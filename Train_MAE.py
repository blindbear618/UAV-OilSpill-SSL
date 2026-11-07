# Train_MAE.py
import os
import glob
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_utils import OilSpillH5Dataset
from tqdm import tqdm


# ==========================================
# 1. 图像分块与位置编码模块
# ==========================================
class PatchEmbed(nn.Module):
    """将 16x16 的多光谱切片划分为 2x2 的 Token"""

    def __init__(self, img_size=16, patch_size=2, in_chans=30, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # 8 * 8 = 64 个 Token

        # 使用卷积进行无重叠切块和降维
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x 形状: (B, 30, 16, 16)
        x = self.proj(x)  # 变为 (B, 256, 8, 8)
        x = x.flatten(2).transpose(1, 2)  # 变为 (B, 64, 256) 序列格式
        return x


# ==========================================
# 2. SS-MAE 核心骨干网络
# ==========================================
class SS_MAE_Core(nn.Module):
    """空间-光谱掩码自编码器"""

    def __init__(self, in_chans=30, embed_dim=256, depth=4, num_heads=8, mask_ratio=0.75, patch_size=2):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.in_chans = in_chans

        # --- 编码器部分 ---
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # --- 解码器部分 ---
        decoder_dim = 128
        self.decoder_embed = nn.Linear(embed_dim, decoder_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, decoder_dim))

        decoder_layer = nn.TransformerEncoderLayer(d_model=decoder_dim, nhead=4, batch_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=2)

        # 预测头
        self.pred_head = nn.Linear(decoder_dim, (patch_size ** 2) * in_chans)

    def patchify(self, imgs):
        """将原始图像转换为可计算 Loss 的序列目标"""
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        h = w = imgs.shape[2] // p

        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * self.in_chans))
        return x

    def forward(self, x):
        B = x.shape[0]

        # 1. 图像切块映射
        x_embed = self.patch_embed(x)
        x_embed = x_embed + self.pos_embed[:, 1:, :]

        # 2. 随机掩码
        L = x_embed.shape[1]
        len_keep = int(L * (1 - self.mask_ratio))

        noise = torch.rand(B, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x_embed, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x_embed.shape[-1]))

        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        # 加入 CLS Token 进入编码器
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(B, -1, -1)
        x_masked = torch.cat((cls_tokens, x_masked), dim=1)

        latent = self.encoder(x_masked)

        # 3. 解码器重构
        latent = self.decoder_embed(latent)

        mask_tokens = self.mask_token.repeat(B, L + 1 - latent.shape[1], 1)
        latent_ = torch.cat([latent[:, 1:, :], mask_tokens], dim=1)
        latent_ = torch.gather(latent_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, latent.shape[2]))
        latent_ = torch.cat([latent[:, :1, :], latent_], dim=1)

        latent_ = latent_ + self.decoder_pos_embed
        decoded = self.decoder(latent_)
        decoded = decoded[:, 1:, :]

        # 4. 预测输出
        pred = self.pred_head(decoded)

        return pred, mask


# ==========================================
# 3. MAE 预训练主循环 (含智能断点续训)
# ==========================================
def train_mae(h5_path, epochs=100, batch_size=1024, learning_rate=5e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OilSpillH5Dataset(h5_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model = SS_MAE_Core(in_chans=30, embed_dim=256, mask_ratio=0.75).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    start_epoch = 0

    # 断点恢复逻辑
    checkpoint_pattern = "weights/mae/mae_checkpoint_ep*.pth"
    checkpoints = glob.glob(checkpoint_pattern)
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"🔄 发现历史存档，正在恢复: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch']
        print(f"✅ 成功从第 {start_epoch} 轮继续训练！")

    print("🚀 启动 SS-MAE 掩码自编码器预训练...")
    start_time = time.time()

    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{epochs}]")

        for images in pbar:
            images = images.to(device)
            pred, mask = model(images)
            target = model.patchify(images)

            loss = (pred - target) ** 2
            loss = loss.mean(dim=-1)
            loss = (loss * mask).sum() / mask.sum()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'MSE Loss': f"{loss.item():.5f}"})

        scheduler.step()  # 步进学习率
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} 结束, 平均重建 Loss: {avg_loss:.5f}, 当前学习率: {scheduler.get_last_lr()[0]:.6e}")

        if (epoch + 1) % 5 == 0:
            os.makedirs("weights/mae2", exist_ok=True)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }
            save_path = f"weights/mae2/mae_checkpoint_ep{epoch + 1}.pth"
            torch.save(checkpoint, save_path)
            print(f"💾 已自动存档: {save_path}")

    print(f"🎉 MAE 预训练完成！自上次启动耗时: {(time.time() - start_time) / 60:.2f} 分钟")

    os.makedirs("weights/mae2", exist_ok=True)
    full_path = "weights/mae2/mae_pretrained_full.pth"
    enc_path = "weights/mae2/mae_pretrained_encoder.pth"
    torch.save(model.state_dict(), full_path)
    torch.save(model.encoder.state_dict(), enc_path)
    print(f"✅ MAE 最终编码器权重已保存至: {enc_path}")


if __name__ == "__main__":
    H5_PATH = r"E:\Dsw\Dataset\OilSpill_Dataset.h5"
    train_mae(H5_PATH)