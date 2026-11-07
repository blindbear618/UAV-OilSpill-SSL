# Train_SimCLR.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_utils import OilSpillH5Dataset
from tqdm import tqdm
import os
import glob
import time


# ==========================================
# 1. 适用于多光谱的小型 CNN 骨干网络
# ==========================================
class SimpleEncoder(nn.Module):
    def __init__(self, in_chans=30, feature_dim=256):
        super().__init__()
        # 16x16 -> 8x8 -> 4x4
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 8x8

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 4x4

            nn.Conv2d(128, feature_dim, kernel_size=4, stride=1, padding=0),  # 1x1
            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)


# ==========================================
# 2. SimCLR 核心架构
# ==========================================
class SimCLR(nn.Module):
    def __init__(self, in_chans=30, feature_dim=256, out_dim=128):
        super().__init__()
        self.encoder = SimpleEncoder(in_chans, feature_dim)
        # 投影头 (Projection Head)
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, out_dim)
        )

    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return z_i, z_j


# ==========================================
# 3. 多光谱专属数据增强
# 注意：多光谱不能用 RGB 的色彩抖动 (ColorJitter)，只能用空间几何变换和加噪
# ==========================================
def multispectral_augment(x):
    x_aug = x.clone()

    # 1. 空间翻转
    if torch.rand(1).item() > 0.5: x_aug = torch.flip(x_aug, dims=[3])
    if torch.rand(1).item() > 0.5: x_aug = torch.flip(x_aug, dims=[2])

    # 2. 💡 随机中心裁剪并放大
    if torch.rand(1).item() > 0.5:
        crop = x_aug[:, :, 2:14, 2:14]  # 截取中间 12x12
        x_aug = F.interpolate(crop, size=(16, 16), mode='bilinear', align_corners=False)

    # 3. 💡 随机遮挡 Cutout
    if torch.rand(1).item() > 0.5:
        if torch.rand(1).item() > 0.5:
            x_aug[:, :, :8, :8] = 0  # 涂黑左上角 1/4
        else:
            x_aug[:, :, 8:, 8:] = 0  # 涂黑右下角 1/4

    # 4. 底噪与通道缺失
    x_aug = x_aug + torch.randn_like(x_aug) * 0.05
    if torch.rand(1).item() > 0.5:
        mask = torch.rand(x_aug.shape[0], x_aug.shape[1], 1, 1, device=x.device) > 0.1
        x_aug = x_aug * mask

    return torch.clamp(x_aug, 0.0, 1.0)


# ==========================================
# 4. NT-Xent 对比损失函数
# ==========================================
def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.shape[0]
    z = torch.cat((z_i, z_j), dim=0)
    z = F.normalize(z, dim=1)

    # 计算余弦相似度矩阵
    sim_matrix = torch.matmul(z, z.T) / temperature

    # 构造标签
    labels = torch.cat([torch.arange(batch_size) for i in range(2)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(z.device)

    # 屏蔽对角线 (自己和自己算相似度没有意义)
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(z.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    sim_matrix = sim_matrix[~mask].view(sim_matrix.shape[0], -1)

    # 提取正样本对
    positives = sim_matrix[labels.bool()].view(labels.shape[0], -1)
    # 负样本是剩下的
    negatives = sim_matrix[~labels.bool()].view(sim_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(z.device)

    loss = F.cross_entropy(logits, labels)
    return loss


def train_simclr(h5_path, epochs=100, batch_size=2048):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OilSpillH5Dataset(h5_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model = SimCLR().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    start_epoch = 0

    checkpoint_pattern = "weights/simclr/simclr_checkpoint_ep*.pth"
    checkpoints = glob.glob(checkpoint_pattern)
    if checkpoints:
        # 按照文件的修改时间找到最新的存档
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"🔄 发现历史存档，正在恢复: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"✅ 成功从第 {start_epoch} 轮继续训练！")

    print("🚀 启动 SimCLR 对比学习预训练...")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{epochs}]")

        for images in pbar:
            images = images.to(device)
            view_1 = multispectral_augment(images)
            view_2 = multispectral_augment(images)

            z_1, z_2 = model(view_1, view_2)
            loss = nt_xent_loss(z_1, z_2)
            z_std = F.normalize(z_1, dim=1).std(dim=0).mean().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'InfoNCE Loss': f"{loss.item():.4f}", 'STD': f"{z_std:.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} 结束, 平均 Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            os.makedirs("weights/simclr", exist_ok=True)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            save_path = f"weights/simclr/simclr_checkpoint_ep{epoch + 1}.pth"
            torch.save(checkpoint, save_path)
            print(f"💾 已自动存档: {save_path}")

    os.makedirs("weights/simclr", exist_ok=True)
    final_path = "weights/simclr/simclr_pretrained.pth"
    torch.save(model.encoder.state_dict(), final_path)
    print(f"✅ SimCLR 预训练全部完成！最终权重已保存至: {final_path}")


if __name__ == "__main__":
    H5_PATH = r"E:\Dsw\Dataset\OilSpill_Dataset.h5"
    train_simclr(H5_PATH)