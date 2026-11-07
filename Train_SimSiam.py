# Train_SimSiam.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset_utils import OilSpillH5Dataset
from tqdm import tqdm
import os
import glob

# 复用刚才写的 CNN 编码器和增强策略 (为保持文件独立，这里重写精简版)
class SimpleEncoder(nn.Module):
    def __init__(self, in_chans=30, feature_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, feature_dim, 4, 1, 0), nn.Flatten()
        )

    def forward(self, x): return self.net(x)


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
# SimSiam 核心架构
# ==========================================
class SimSiam(nn.Module):
    def __init__(self, in_chans=30, feature_dim=256, proj_dim=128):
        super().__init__()
        self.encoder = SimpleEncoder(in_chans, feature_dim)

        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False)
        )

        # 预测头 (Predictor) - SimSiam 的灵魂
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, proj_dim // 2, bias=False),
            nn.BatchNorm1d(proj_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim // 2, proj_dim)
        )

    def forward(self, x1, x2):
        z1, z2 = self.projector(self.encoder(x1)), self.projector(self.encoder(x2))
        p1, p2 = self.predictor(z1), self.predictor(z2)
        # 返回预测值和目标值 (目标值会截断梯度)
        return p1, p2, z1.detach(), z2.detach()


# ==========================================
# 负余弦相似度损失
# ==========================================
def D(p, z):
    # 归一化后点乘
    z = z.detach()  # 再次确认梯度截断
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return -(p * z).sum(dim=1).mean()


def train_simsiam(h5_path, epochs=100, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OilSpillH5Dataset(h5_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model = SimSiam().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    start_epoch = 0

    checkpoint_pattern = "weights/simsiam/simsiam_checkpoint_ep*.pth"
    checkpoints = glob.glob(checkpoint_pattern)
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"🔄 发现历史存档，正在恢复: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"✅ 成功从第 {start_epoch} 轮继续训练！")

    print("🚀 启动 SimSiam 非对比学习预训练...")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{epochs}]")

        for images in pbar:
            images = images.to(device)
            view_1, view_2 = multispectral_augment(images), multispectral_augment(images)

            p1, p2, z1, z2 = model(view_1, view_2)
            loss = D(p1, z2) / 2 + D(p2, z1) / 2

            z_std = F.normalize(z1, dim=1).std(dim=0).mean().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'STD': f"{z_std:.4f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} 结束, 平均 Loss: {avg_loss:.4f}")

        if (epoch + 1) % 5 == 0:
            os.makedirs("weights/simsiam", exist_ok=True)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            save_path = f"weights/simsiam/simsiam_checkpoint_ep{epoch + 1}.pth"
            torch.save(checkpoint, save_path)
            print(f"💾 已自动存档: {save_path}")

    os.makedirs("weights/simsiam", exist_ok=True)
    final_path = "weights/simsiam/simsiam_pretrained.pth"
    torch.save(model.encoder.state_dict(), final_path)
    print(f"✅ SimSiam 预训练全部完成！最终权重已保存至: {final_path}")


if __name__ == "__main__":
    H5_PATH = r"E:\Dsw\Dataset\OilSpill_Dataset.h5"
    train_simsiam(H5_PATH)