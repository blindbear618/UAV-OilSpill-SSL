# Train_DAE.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset_utils import OilSpillH5Dataset
from tqdm import tqdm
import os
import glob

# ==========================================
# 降噪自编码器 (Denoising AutoEncoder)
# ==========================================
class DAE(nn.Module):
    def __init__(self, in_chans=30):
        super().__init__()
        # 编码器: 16x16 -> 8x8 -> 4x4
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans, 64, 3, stride=2, padding=1),  # 8x8
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 4x4
            nn.ReLU()
        )
        # 解码器: 4x4 -> 8x8 -> 16x16
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, in_chans, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 反射率被规约在 0~1 之间
        )

    def forward(self, x):
        features = self.encoder(x)
        reconstruction = self.decoder(features)
        return reconstruction, features


def train_dae(h5_path, epochs=100, batch_size=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = OilSpillH5Dataset(h5_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

    model = DAE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
    criterion = nn.MSELoss()

    start_epoch = 0

    checkpoint_pattern = "weights/dae/dae_checkpoint_ep*.pth"
    checkpoints = glob.glob(checkpoint_pattern)
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        print(f"🔄 发现历史存档，正在恢复: {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"✅ 成功从第 {start_epoch} 轮继续训练！")

    print("🚀 启动 DAE 降噪自编码器预训练...")
    for epoch in range(start_epoch, epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{epochs}]")

        for images in pbar:
            images = images.to(device)
            noise = torch.randn_like(images) * 0.1
            noisy_images = images + noise
            if torch.rand(1).item() > 0.5:
                mask = torch.rand(images.shape[0], images.shape[1], 1, 1, device=device) > 0.1
                noisy_images = noisy_images * mask
            noisy_images = torch.clamp(noisy_images, 0.0, 1.0)

            reconstructed, _ = model(noisy_images)
            loss = criterion(reconstructed, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'MSE Loss': f"{loss.item():.5f}"})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1} 结束, 平均 Loss: {avg_loss:.5f}")

        if (epoch + 1) % 5 == 0:
            os.makedirs("weights/dae", exist_ok=True)
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }
            save_path = f"weights/dae/dae_checkpoint_ep{epoch + 1}.pth"
            torch.save(checkpoint, save_path)
            print(f"💾 已自动存档: {save_path}")

    os.makedirs("weights/dae", exist_ok=True)
    final_path = "weights/dae/dae_pretrained.pth"
    torch.save(model.encoder.state_dict(), final_path)
    print(f"✅ DAE 预训练全部完成！最终权重已保存至: {final_path}")


if __name__ == "__main__":
    H5_PATH = r"E:\Dsw\Dataset\OilSpill_Dataset.h5"
    train_dae(H5_PATH)