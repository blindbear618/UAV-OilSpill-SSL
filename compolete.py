# -*- coding: utf-8 -*-
"""
compolete.py
修复内容（相对于 最终代码.py）：
  [修复1] MAE 权重加载：支持完整 checkpoint（mae_pretrained_full.pth，键名原样匹配）
          与旧的 encoder-only 权重（自动补 'encoder.' 前缀），并过滤 decoder 冗余键
  [修复2] DAE_Encoder 架构改回与预训练完全一致（两层 stride=2 卷积），fc 为新增层
  [修复3] 算法路由兼容 Windows 反斜杠路径，且同时检查目录名与文件名
  [修复4] 权重加载改为智能多映射匹配 + 加载张量计数 + 零匹配直接报错，杜绝静默失败
  [修复5] 波段重要性绘图支持 MAE（patch_embed.proj）与 CNN 两种编码器，不再崩溃
  [修复6] 数据划分改为"类别内顺序块划分"：每类内部按原始顺序前 80% 训练、后 20% 验证
          —— HDF5 中同类 patch 按"源影像 × 光栅扫描"顺序存储，块划分近似等价于
          按源影像隔离，既修复了顺序划分导致的类别失衡灾难，又把重叠泄漏压缩到
          类别内部切分缝处的极少数 patch。彻底方案见 run_pipeline 内注释。
  [修复7] t-SNE 仅在验证集上可视化（原来用了含训练集的全量数据）

扩展实验（投稿前补充，v2 新增）：
  [扩展1] set_seed + finetune_once：单次微调核心函数，每个种子独立保存最佳 pth
  [扩展2] run_multiseed_experiment：3 种子微调 → mean±std（论文主表格式）
  [扩展3] run_block_kfold：类内块级 5 折交叉验证（稳健性附录）
  [扩展4] run_classical_baselines：SVM-RBF / 随机森林基线（30 维平均光谱，共用同一划分）
          run_linear_probe：冻结编码器只训线性头（SSL 标准协议）
          finetune_once(weight_path=None, algo=...)：同架构从头训练基线
  [扩展5] threshold_sensitivity_analysis：0.55 置信门限 / 0.25 NIR 否决的敏感性扫描
  [扩展6] cell_level_accuracy：10 个围油栏网格的多数投票定量评估（实景图定量证据）
  [扩展7] run_scratch_baseline：从头训练基线的多种子包装（4架构×3种子）
          run_all_experiments：一键总控——数据集只加载一次，日志安全重定向，
          每阶段独立落盘 + 异常隔离，适合整夜连跑；finetune_once 新增 encoder_lr/head_lr
          参数与显存主动释放
  [扩展8] 性能修复：set_seed 默认 cudnn_fast=True（cudnn.benchmark 自动调优）——
          deterministic=True 曾导致卷积慢 4-5 倍，多种子实验无需逐位可复现；
          全部 DataLoader 加 pin_memory（num_workers 保持 0，避免 fork 复制全量内存数据集）
  [扩展9] run_kfold_all：四模型块级 K 折一键连跑（论文主表方案——无泄漏且验证块
          覆盖全部时间区段，均值代表平均难度）；check_block_shift / random_split_diagnostic：
          训练/验证块光谱漂移诊断与随机划分对照（诊断结果不进论文）
  [扩展10] run_label_efficiency_experiment：标签效率实验（100%/25%/10%/5%/1% 标签，
          预训练 vs 从头训练对比）——全量标签下预训练未必占优，少标签场景才是 SSL
          价值的关键验证；类别均衡+时间均匀抽样，产出 CSV 与论文曲线图
"""
import os
import sys
import time
import datetime
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from dataset_utils import OilSpill4ClassDataset
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from itertools import cycle
import spectral.io.envi as envi
from scipy.signal import savgol_filter

# ==========================================
# 0. 全局设置与日志工具 (SCI 标准)
# ==========================================
plt.rcParams.update({
    "font.family": 'serif',
    "font.serif": ['Times New Roman'],
    "font.size": 14,
    "axes.unicode_minus": False
})


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        pass


_ORIGINAL_STDOUT = sys.stdout  # 真实终端句柄，供日志重定向前恢复

def _reset_logger(log_path):
    """重定向 stdout 到日志文件。先恢复到真实终端再包 Logger，
    避免同一轮连跑多个实验函数时 Logger 嵌套串链、日志互相污染。"""
    sys.stdout = _ORIGINAL_STDOUT
    sys.stdout = Logger(log_path)


# ==========================================
# 1. 损失函数与模型结构
# ==========================================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
        return focal_loss.mean()


class PatchEmbed(nn.Module):
    def __init__(self, img_size=16, patch_size=2, in_chans=30, embed_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class MAE_Encoder(nn.Module):
    """MAE 下游编码器：键名设计与 SS_MAE_Core 完整 state_dict 的编码器部分严格一致
    （patch_embed.* / cls_token / pos_embed / encoder.layers.*），
    因此可直接加载 mae_pretrained_full.pth。"""

    def __init__(self, in_chans=30, embed_dim=256, depth=4, num_heads=8):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        B = x.shape[0]
        x_embed = self.patch_embed(x) + self.pos_embed[:, 1:, :]
        cls_tokens = (self.cls_token + self.pos_embed[:, :1, :]).expand(B, -1, -1)
        x_concat = torch.cat((cls_tokens, x_embed), dim=1)
        latent = self.encoder(x_concat)
        return latent[:, 0, :]  # CLS token 作为全局特征 (B, 256)


class DAE_Encoder(nn.Module):
    """[修复2] 与 Train_DAE.py 的预训练编码器结构逐层一致：
    Conv(30->64, s=2) -> ReLU -> Conv(64->128, s=2) -> ReLU，输出 4x4x128。
    fc 为下游新增层（随机初始化属正常，配合差分学习率训练）。"""

    def __init__(self, in_chans=30, feature_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 64, 3, stride=2, padding=1), nn.ReLU(),   # 16x16 -> 8x8
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU()          # 8x8 -> 4x4
        )
        self.fc = nn.Linear(128 * 4 * 4, feature_dim)

    def forward(self, x):
        return self.fc(torch.flatten(self.net(x), 1))


class SimpleEncoder(nn.Module):
    """SimCLR / SimSiam 下游 CNN 架构（键名 net.* / fc.*，与预训练 encoder.net.* 对应）"""

    def __init__(self, in_chans=30, feature_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 4, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.fc = nn.Linear(256, feature_dim)

    def forward(self, x):
        return self.fc(torch.flatten(self.net(x), 1))


# ==========================================
# 1.5 [修复3/4] 算法识别与智能权重加载
# ==========================================
def detect_algo(weight_path):
    """兼容 Windows 反斜杠路径；同时检查目录名与文件名，避免误判。"""
    norm = weight_path.replace('\\', '/')
    parts = norm.split('/')
    folder = parts[-2].upper() if len(parts) >= 2 else ''
    fname = parts[-1].upper()
    for tag in ('SIMSIAM', 'SIMCLR', 'MAE', 'DAE'):  # 先长后短，防止 SIMCLR 命中 SIM 前缀歧义
        if tag in folder or tag in fname:
            return tag
    return 'UNKNOWN'


def smart_load_encoder(encoder, state_dict, algo_name):
    """[修复4] 尝试多种键名映射，自动选择匹配张量数最多的一种；
    零匹配直接抛错，绝不再静默随机初始化。"""
    target_sd = encoder.state_dict()

    def t_identity(sd):
        return dict(sd)

    def t_strip_encoder(sd):  # SimCLR/SimSiam: encoder.net.* -> net.*
        return {k[len('encoder.'):]: v for k, v in sd.items() if k.startswith('encoder.')}

    def t_add_encoder(sd):    # 旧的 MAE encoder-only 权重: layers.* -> encoder.layers.*
        return {f'encoder.{k}': v for k, v in sd.items()}

    def t_add_net(sd):        # DAE encoder-only 权重: 0.weight -> net.0.weight
        return {f'net.{k}': v for k, v in sd.items()}

    candidates = {
        'identity': t_identity,
        'strip_encoder': t_strip_encoder,
        'add_encoder': t_add_encoder,
        'add_net': t_add_net,
    }

    best_name, best_sd, best_matched = None, None, -1
    for name, fn in candidates.items():
        cand = fn(state_dict)
        matched = sum(1 for k, v in cand.items()
                      if k in target_sd and target_sd[k].shape == v.shape)
        if matched > best_matched:
            best_name, best_sd, best_matched = name, cand, matched

    # 只保留形状严格一致的键
    load_sd = {k: v for k, v in best_sd.items()
               if k in target_sd and target_sd[k].shape == v.shape}
    missing, unexpected = encoder.load_state_dict(load_sd, strict=False)

    print(f"   -> 键映射方式: {best_name} | 成功加载 {len(load_sd)} 个参数张量")
    if missing:
        print(f"   -> 未初始化层 {len(missing)} 个（新增层随机初始化属正常）: {missing[:5]}")
    if unexpected:
        print(f"   -> 忽略冗余键 {len(unexpected)} 个（如 decoder/projector）: {unexpected[:5]}")

    if len(load_sd) == 0:
        raise RuntimeError(
            f"❌ 预训练权重一个参数都没加载上（识别架构: {algo_name}）！"
            f"请检查权重文件路径与编码器结构是否对应。")

    # MAE 特检：若 cls_token/pos_embed 缺失，说明喂的是 encoder-only 旧权重，提醒换 full checkpoint
    if algo_name == 'MAE':
        critical = [k for k in ('cls_token', 'pos_embed') if k in missing]
        if critical:
            print(f"   ⚠️ 注意: {critical} 未加载（当前权重可能是 encoder-only 旧文件），"
                  f"建议改用 mae_pretrained_full.pth 以获得完整位置编码。")
    return best_name, len(load_sd)


class FineTuneModel(nn.Module):
    """下游分类模型。
    用法A（默认）：FineTuneModel(weight_path='路径.pth') —— 自动识别架构并加载预训练权重
    用法B（基线）：FineTuneModel(weight_path=None, algo='MAE') —— 同架构从头训练（随机初始化），
                  用于量化 SSL 预训练的净收益。"""

    def __init__(self, weight_path=None, num_classes=4, algo=None):
        super().__init__()
        if weight_path is not None:
            algo_name = detect_algo(weight_path)  # [修复3]
        else:
            algo_name = (algo or '').upper()
            assert algo_name in ('MAE', 'DAE', 'SIMCLR', 'SIMSIAM'), \
                "从头训练模式必须显式指定 algo ∈ {MAE, DAE, SIMCLR, SIMSIAM}"

        if algo_name == "MAE":
            self.encoder = MAE_Encoder()
        elif algo_name == "DAE":
            self.encoder = DAE_Encoder()
        elif algo_name in ("SIMCLR", "SIMSIAM"):
            self.encoder = SimpleEncoder()
        else:
            print(f"   ⚠️ 未能从路径识别算法类型，默认使用 SimpleEncoder: {weight_path}")
            self.encoder = SimpleEncoder()

        if weight_path is not None:
            print(f"📦 正在载入预训练权重: {weight_path} (识别架构: {algo_name})")
            state_dict = torch.load(weight_path, map_location='cpu')
            # [修复4b] 兼容训练中途存档的包装格式 {'model_state_dict': ...}（Train_MAE.py 的 epoch checkpoint）
            if isinstance(state_dict, dict):
                for wrap_key in ('model_state_dict', 'state_dict', 'model'):
                    if wrap_key in state_dict and isinstance(state_dict[wrap_key], dict):
                        print(f"   -> 检测到 checkpoint 包装键 '{wrap_key}'，自动解包")
                        state_dict = state_dict[wrap_key]
                        break
            smart_load_encoder(self.encoder, state_dict, algo_name)  # [修复4]
            print(f"   -> 基础参数载入完毕！")
        else:
            print(f"🎲 从头训练基线模式: {algo_name} 架构，全部参数随机初始化（不加载预训练权重）")

        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.encoder(x))


# ==========================================
# 2. 自动化绘图函数大礼包
# ==========================================
def draw_spectral_signatures(full_dataset, save_dir, num_samples_per_class=1000):
    print("\n📈 开始提取并绘制原始光谱反射率曲线...")
    class_names = ['Pure Water', 'Gasoline', 'Diesel', 'Mineral Oil']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    spectra_data = {i: [] for i in range(4)}
    counts = {i: 0 for i in range(4)}

    np.random.seed(42)
    indices = np.random.permutation(len(full_dataset))

    for idx in indices:
        data_tensor, label = full_dataset[idx]
        label = int(label)

        if counts[label] < num_samples_per_class:
            mean_spectrum = data_tensor.numpy().mean(axis=(1, 2))
            spectra_data[label].append(mean_spectrum)
            counts[label] += 1

        if all(c >= num_samples_per_class for c in counts.values()):
            break

    plt.figure(figsize=(10, 6), dpi=300)
    wavelengths = np.linspace(400, 1000, 30)

    for i in range(4):
        data_arr = np.array(spectra_data[i])
        mean_spec = np.mean(data_arr, axis=0)
        std_spec = np.std(data_arr, axis=0)
        plt.plot(wavelengths, mean_spec, label=class_names[i], color=colors[i], linewidth=2.5)
        plt.fill_between(wavelengths, mean_spec - std_spec, mean_spec + std_spec, color=colors[i], alpha=0.15)

    plt.title('Normalized Spectral Signatures of Oil Spills & Water', fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
    plt.ylabel('Normalized Reflectance', fontsize=14, fontweight='bold')
    plt.xticks(np.arange(400, 1001, 100))
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "01_Spectral_Signatures.pdf"))
    plt.savefig(os.path.join(save_dir, "01_Spectral_Signatures.png"))
    plt.close()
    print("🎨 [1/6] 光谱特征折线图绘制完成！")


def draw_sci_confusion_matrix(cm, classes, save_dir):
    print("\n🧮 开始绘制混淆矩阵热力图...")
    cm_perc = cm / np.sum(cm, axis=1, keepdims=True) * 100
    plt.figure(figsize=(8, 6), dpi=300)
    ax = sns.heatmap(cm_perc, annot=False, cmap='YlGnBu', fmt=".1f",
                     linewidths=.5, linecolor='black', vmin=0, vmax=100,
                     cbar_kws={'label': 'Accuracy (%)'})

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm_perc[i, j] > 50 else "black"
            ax.text(j + 0.5, i + 0.5, f"{cm_perc[i, j]:.1f}%\n({cm[i, j]})",
                    ha="center", va="center", color=text_color, fontsize=12, fontweight='bold')

    tick_marks = np.arange(len(classes)) + 0.5
    plt.xticks(tick_marks, classes, rotation=15, fontsize=12)
    plt.yticks(tick_marks, classes, rotation=0, fontsize=12)
    plt.ylabel('True Class', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=15)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "02_Confusion_Matrix.pdf"))
    plt.savefig(os.path.join(save_dir, "02_Confusion_Matrix.png"))
    plt.close()
    print("🎨 [2/6] 混淆矩阵绘制完成！")


def draw_roc_curve(all_labels, all_probs, save_dir):
    print("\n📐 开始绘制多分类 ROC 曲线与 AUC...")
    y_test_bin = label_binarize(all_labels, classes=[0, 1, 2, 3])
    n_classes = y_test_bin.shape[1]

    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    plt.figure(figsize=(9, 8), dpi=300)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guessing')
    plt.plot(all_fpr, mean_tpr, label=f'Macro-average ROC (AUC = {auc(all_fpr, mean_tpr):.4f})',
             color='black', linestyle=':', linewidth=3)

    class_names = ['Pure Water', 'Gasoline', 'Diesel', 'Mineral Oil']
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    for i, color, name in zip(range(n_classes), colors, class_names):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, alpha=0.8, label=f'{name} (AUC = {roc_auc[i]:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    plt.title('ROC Curves', fontsize=16, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=12, frameon=True, edgecolor='black')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "03_ROC_Curves.pdf"))
    plt.savefig(os.path.join(save_dir, "03_ROC_Curves.png"))
    plt.close()
    print("🎨 [3/6] ROC 曲线绘制完成！")


def draw_tsne(model, eval_dataset, device, save_dir):
    """[修复7] 仅在验证集上取样可视化，避免把训练集特征画进论文图。"""
    print("\n🧠 开始执行 t-SNE 特征空间降维可视化（仅验证集）...")
    model.eval()

    np.random.seed(42)
    n_pick = min(4000, len(eval_dataset))
    subset_indices = np.random.choice(len(eval_dataset), n_pick, replace=False)
    dataset = Subset(eval_dataset, subset_indices)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    all_features, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = model.encoder(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    tsne = TSNE(n_components=2, perplexity=40, random_state=42)
    tsne_results = tsne.fit_transform(all_features)

    plt.figure(figsize=(10, 8), dpi=300)
    class_names = ['Pure Water', 'Gasoline', 'Diesel', 'Mineral Oil']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, name in enumerate(class_names):
        idx = (all_labels == i)
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], c=colors[i], label=name, alpha=0.7, s=15,
                    edgecolors='none')

    plt.legend(markerscale=2, fontsize=12, loc='best')
    plt.title("t-SNE Feature Space", fontsize=16, fontweight='bold', pad=15)
    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "04_tSNE_Features.pdf"))
    plt.savefig(os.path.join(save_dir, "04_tSNE_Features.png"))
    plt.close()
    print("🎨 [4/6] t-SNE 降维特征图绘制完成！")


def draw_band_importance(model, save_dir, in_chans=30):
    """[修复5] 兼容 MAE（patch_embed.proj）与 CNN（net[0]）两种首层结构。"""
    print("\n🔍 开始分析并绘制波段特征重要性贡献图...")
    enc = model.encoder
    if hasattr(enc, 'patch_embed'):            # MAE: Conv2d(30, 256, k=2) -> [256, 30, 2, 2]
        first_layer_weights = enc.patch_embed.proj.weight.data.cpu().numpy()
    elif hasattr(enc, 'net'):                  # CNN 系列: net[0] -> [64, 30, 3, 3]
        first_layer_weights = enc.net[0].weight.data.cpu().numpy()
    else:
        print("   ⚠️ 未知编码器结构，跳过波段重要性分析。")
        return

    importance = np.sum(np.abs(first_layer_weights), axis=(0, 2, 3))
    importance_percent = (importance / np.sum(importance)) * 100

    wavelengths = np.linspace(400, 1000, in_chans)

    plt.figure(figsize=(12, 6), dpi=300)
    colors = plt.cm.viridis(importance_percent / max(importance_percent))
    bars = plt.bar(wavelengths, importance_percent, width=15, color=colors, edgecolor='black', linewidth=0.8)

    top_3_idx = np.argsort(importance_percent)[-3:]
    for idx in top_3_idx:
        bars[idx].set_color('#d62728')
        bars[idx].set_edgecolor('black')
        plt.text(wavelengths[idx], importance_percent[idx] + 0.2, f'Top {3 - list(top_3_idx).index(idx)}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color='#d62728')

    plt.title('Relative Importance of Spectral Wavelengths Learned by the Encoder',
              fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Wavelength (nm)', fontsize=14, fontweight='bold')
    plt.ylabel('Contribution Score (%)', fontsize=14, fontweight='bold')
    plt.xticks(np.arange(400, 1001, 100))
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "05_Band_Importance.pdf"))
    plt.savefig(os.path.join(save_dir, "05_Band_Importance.png"))
    plt.close()
    print("🎨 [5/6] 波段重要性柱状图绘制完成！")


def draw_sliding_window_heatmap(model, device, test_hdr_path, save_dir):
    print("\n🗺️ 开始执行全图滑动窗口推理解译 (带物理掩膜与全局拉伸保护)...")
    model.eval()

    envi_dat_path = test_hdr_path.replace('.hdr', '.dat')

    try:
        print(f"   -> 正在读取测试高光谱影像: {test_hdr_path}")
        img_open = envi.open(test_hdr_path, envi_dat_path)
        raw_image = np.array(img_open.load(), dtype=np.float32)
    except Exception as e:
        print(f"   ❌ 读取影像失败: {e}。全图绘制跳过。")
        return

    print("   -> 正在执行物理清洗与平滑滤波...")
    nir_band_idx = 29
    target_nir = 0.01
    glint_corrected = np.copy(raw_image)
    if raw_image.shape[2] > nir_band_idx:
        nir_band = raw_image[:, :, nir_band_idx]
        delta_glint = nir_band - target_nir
        delta_glint[delta_glint < 0] = 0
        for i in range(raw_image.shape[2]):
            glint_corrected[:, :, i] -= delta_glint
        glint_corrected = np.clip(glint_corrected, 0, 1)
    else:
        glint_corrected = np.clip(raw_image, 0, 1)

    if glint_corrected.shape[0] >= 5 and glint_corrected.shape[1] >= 5:
        smoothed_image = savgol_filter(glint_corrected, window_length=5, polyorder=2, axis=2)
    else:
        smoothed_image = glint_corrected

    # =========================================================
    # 🛡️ 提取光学物理掩膜 (围油栏与背景拦截)
    # =========================================================
    green_band_raw = smoothed_image[:, :, 7]
    nir_band_raw = smoothed_image[:, :, 29]
    ndwi_raw = (green_band_raw - nir_band_raw) / (green_band_raw + nir_band_raw + 1e-8)

    # 塑料边框 NDWI < 0，或 NIR 异常高反(>0.25)
    background_mask = (ndwi_raw < 0.0) | (nir_band_raw > 0.25)

    large_image_hwc = smoothed_image[:, :, :30]

    # 🔥 全局 Min-Max 绝对拉伸对齐
    TRAIN_GLOBAL_MIN = -0.2286
    TRAIN_GLOBAL_MAX = 1.1117

    if TRAIN_GLOBAL_MAX - TRAIN_GLOBAL_MIN > 1e-6:
        large_image_hwc = (large_image_hwc - TRAIN_GLOBAL_MIN) / (TRAIN_GLOBAL_MAX - TRAIN_GLOBAL_MIN)
        large_image_hwc = np.clip(large_image_hwc, 0, 1)

    img_h, img_w, _ = large_image_hwc.shape
    print(f"   -> 影像尺寸: {img_w}x{img_h}x30, 开始滑动窗口推理...")

    patch_size = 16
    stride = 8

    prob_map = np.zeros((4, img_h, img_w), dtype=np.float32)
    count_map = np.zeros((img_h, img_w), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, img_h - patch_size + 1, stride):
            for x in range(0, img_w - patch_size + 1, stride):
                patch_hwc = large_image_hwc[y:y + patch_size, x:x + patch_size, :]
                if patch_hwc.shape != (16, 16, 30):
                    continue

                patch_chw = np.transpose(patch_hwc, (2, 0, 1))
                patch_tensor = torch.tensor(patch_chw, dtype=torch.float32).unsqueeze(0).to(device)

                logits = model(patch_tensor)
                probs = F.softmax(logits, dim=1).cpu().numpy()[0]

                for c in range(4):
                    prob_map[c, y:y + patch_size, x:x + patch_size] += probs[c]
                count_map[y:y + patch_size, x:x + patch_size] += 1

    count_map[count_map == 0] = 1
    prob_map = prob_map / count_map

    max_prob_map = np.max(prob_map, axis=0)
    final_pred_map = np.argmax(prob_map, axis=0)

    # =========================================================
    # 🎯 终极融合：AI 决策 0.55置信度 + 物理光学一票否决
    # =========================================================
    confidence_threshold = 0.55
    final_pred_map[max_prob_map < confidence_threshold] = 4
    final_pred_map[background_mask] = 4

    from matplotlib.colors import ListedColormap
    import matplotlib.patches as mpatches

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#000000']
    cmap = ListedColormap(colors)

    plt.figure(figsize=(10, 10), dpi=300)
    plt.imshow(final_pred_map, cmap=cmap, interpolation='nearest', vmin=0, vmax=4)

    labels = ['Pure Water', 'Gasoline', 'Diesel', 'Mineral Oil', 'Unknown / Background']
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(5)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    plt.title('Real-scene Sliding Window Inference Heatmap', fontsize=16, fontweight='bold', pad=15)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "06_Sliding_Window_Heatmap.pdf"))
    plt.savefig(os.path.join(save_dir, "06_Sliding_Window_Heatmap.png"))
    plt.close()
    print("🎨 [6/6] 真实场景推理解译热力图绘制完成！")

    # 返回推理中间结果，供网格级精度评估（cell_level_accuracy）等后续分析直接使用
    return {
        'pred_map': final_pred_map,          # (H, W) int: 0-3 四类, 4=背景/未知
        'prob_map': prob_map,                # (4, H, W) 平均后类别概率
        'background_mask': background_mask,  # (H, W) bool: 物理否决区域
        'ndwi': ndwi_raw,                    # (H, W) float
        'nir': nir_band_raw,                 # (H, W) float
    }


# ==========================================
# 2.5 [修复6] 类别内顺序块划分（近似影像级隔离）
# ==========================================
def stratified_block_split(dataset, train_ratio=0.8):
    """
    修复要点：
    1) 全局顺序划分（旧代码）在"按类拼接"的数据集上会造成验证集几乎全是矿物油
       —— 类别失衡灾难。本函数在每个类别内部独立划分，保证四类比例一致。
    2) HDF5 中同一类别的 patch 按"源影像逐文件 × 影像内光栅扫描(步长8)"顺序追加，
       因此类别内的连续块大致对应"不同的源影像/不同空间区域"。类别内顺序块划分
       近似实现了影像级（scene-level）隔离，重叠泄漏被压缩到切分缝处的少量 patch。
    ⚠️ 彻底方案：重建 HDF5 时为每个 patch 记录源影像 ID，再用 GroupShuffleSplit
       按影像 ID 分组划分；或以 stride=16 无重叠切片重建数据库后随机划分。
       在论文中请如实声明所采用的划分策略。
    """
    labels = np.asarray(dataset.ram_labels)
    train_idx, val_idx = [], []
    print("\n📊 类别内顺序块划分（每类前 {:.0f}% 训练 / 后 {:.0f}% 验证）：".format(
        train_ratio * 100, (1 - train_ratio) * 100))
    class_names = ['Pure Water', 'Gasoline', 'Diesel', 'Mineral Oil']
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]          # 保持原始顺序 = 源影像×光栅顺序
        n_train = int(len(idx) * train_ratio)
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train:].tolist())
        print(f"   -> [{class_names[c]}]: 训练 {n_train} / 验证 {len(idx) - n_train}")
    return train_idx, val_idx


# ==========================================
# 3. 核心大流水线 (训练 + 评估)
# ==========================================
def run_pipeline(weight_path, h5_path, test_hdr_path, epochs=50, batch_size=1024):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    algo_name = detect_algo(weight_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"results_{algo_name}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)

    log_file = os.path.join(out_dir, "training_log.txt")
    _reset_logger(log_file)
    print(f"📁 启动终极自动化科研流水线，所有产出将保存至: {out_dir}\n")

    full_dataset = OilSpill4ClassDataset(h5_path)
    draw_spectral_signatures(full_dataset, out_dir)

    # [修复6] 类别内顺序块划分，替代旧的全局顺序划分 / 随机划分
    train_idx, val_idx = stratified_block_split(full_dataset, train_ratio=0.8)
    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)
    print(f"\n📊 划分完成: 训练集 {len(train_dataset)} 张, 验证集 {len(val_dataset)} 张（四类比例一致）")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                            pin_memory=torch.cuda.is_available())

    model = FineTuneModel(weight_path=weight_path, num_classes=4).to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-4},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_weight_save_path = os.path.join(out_dir, f"{algo_name}_best_finetuned.pth")

    print("\n🚀 === 第一阶段：全网络非线性微调 (Focal Loss) ===")
    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        train_acc = 100.0 * train_correct / train_total

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        val_acc = 100.0 * val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weight_save_path)
            mark = " 🏆 (New Best!)"
        else:
            mark = ""

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(
            f"Epoch [{epoch + 1}/{epochs}] | LR: {current_lr:.2e} | Train Loss: {train_loss / len(train_loader):.4f} | Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}%{mark}")

    print("\n🔬 === 第二阶段：自动生成论文评估报告与 6 大核心插图 ===")
    print(f"📦 正在重新加载历史最佳权重...")
    model.load_state_dict(torch.load(best_weight_save_path))
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)

            probs = F.softmax(logits, dim=1)
            all_probs.extend(probs.cpu().numpy())

            _, predicted = torch.max(logits, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_labels_np = np.array(all_labels)
    all_probs_np = np.array(all_probs)
    all_preds_np = np.array(all_preds)

    target_names = ['Pure Water', 'Gasoline', 'Diesel', 'Mineral Oil']
    print("\n📊 --- 最终模型分类报告 (Classification Report) ---")
    print(classification_report(all_labels_np, all_preds_np, target_names=target_names, digits=4))

    cm = confusion_matrix(all_labels_np, all_preds_np)

    draw_sci_confusion_matrix(cm, target_names, save_dir=out_dir)
    draw_roc_curve(all_labels_np, all_probs_np, save_dir=out_dir)
    draw_tsne(model, val_dataset, device, save_dir=out_dir)   # [修复7] 仅验证集
    draw_band_importance(model, save_dir=out_dir)             # [修复5] 兼容 MAE

    # 🌟 调用包含绝对拉伸与光学掩膜的最强推理引擎
    draw_sliding_window_heatmap(model, device, test_hdr_path, save_dir=out_dir)

    print(f"\n🎉 运行结束！所有产出（包含 6 种核心图表的 PDF/PNG 及完整日志）均已打包保存在文件夹: {out_dir}")


# ==========================================
# 4. 可复现性工具：统一随机种子
# ==========================================
def set_seed(seed=42, cudnn_fast=True):
    """固定所有随机源（论文多种子实验的基础）。
    cudnn_fast=True（默认）：开启 cudnn.benchmark 自动调优，卷积走最快算法。
      代价是不保证跨运行的逐位一致（结果只在统计等价意义上可复现）——
      多种子实验本来就要报告 mean±std，逐位一致没有意义，4-5倍的速度损失不值得。
    cudnn_fast=False：确定性算法，逐位可复现但卷积慢数倍，仅调 bug 时用。
    注意：cudnn 标志只影响 GPU 卷积；CPU 运行不受影响。"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if cudnn_fast:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ==========================================
# 5. 单次微调（多种子/交叉验证/基线共用的核心训练函数）
# ==========================================
def finetune_once(full_dataset, train_idx, val_idx, weight_path=None, algo=None,
                  seed=42, epochs=50, batch_size=1024, out_dir=None, tag='',
                  encoder_lr=1e-4, head_lr=1e-3):
    """
    在指定的 (train_idx, val_idx) 划分上完成一次完整微调，返回最终指标字典。
    - weight_path 给定时加载预训练权重；为 None 时必须传 algo，表示同架构从头训练基线。
    - 每个 (seed, tag) 组合保存一个独立的最佳权重 pth，互不覆盖。
    - encoder_lr/head_lr 默认沿用论文统一协议（1e-4/1e-3）。从头训练基线可考虑把
      encoder_lr 提到 1e-3（随机初始化的编码器用 1e-4 可能欠训练，两种都跑并在论文中
      如实报告所选协议最稳妥）。
    """
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size,
                              shuffle=True, num_workers=0,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size,
                            shuffle=False, num_workers=0,
                            pin_memory=torch.cuda.is_available())

    model = FineTuneModel(weight_path=weight_path, num_classes=4, algo=algo).to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = torch.optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': encoder_lr},
        {'params': model.classifier.parameters(), 'lr': head_lr}
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_path = None
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        model_tag = tag or (detect_algo(weight_path) if weight_path else f"{algo}_scratch")
        best_path = os.path.join(out_dir, f"{model_tag}_seed{seed}_best.pth")

    print(f"\n🚀 微调启动 | 模型: {weight_path or (algo + ' 从头训练')} | seed={seed} | "
          f"训练 {len(train_idx)} / 验证 {len(val_idx)}")
    # ===== 性能自检：把关键状态打进日志，方便定位变慢原因 =====
    if torch.cuda.is_available():
        print(f"⚙️  设备: {torch.cuda.get_device_name(0)} | 显存占用: "
              f"{torch.cuda.memory_allocated() / 2**30:.1f} / {torch.cuda.get_device_properties(0).total_memory / 2**30:.0f} GiB")
    print(f"⚙️  cudnn.benchmark = {torch.backends.cudnn.benchmark} | "
          f"cudnn.deterministic = {torch.backends.cudnn.deterministic} "
          f"（应为 True/False；若为 False/True 说明跑的是旧版文件，卷积会慢 4-5 倍）")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct, train_total = 0, 0
        ep_t0 = time.time()
        diag_fetch, diag_step, diag_n = 0.0, 0.0, 0   # epoch 0 前 20 个 batch 的耗时分解
        for images, labels in train_loader:
            t_fetch = time.time()
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            t_step = time.time()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            if epoch == 0 and diag_n < 20:
                diag_fetch += t_step - t_fetch
                diag_step += time.time() - t_step
                diag_n += 1
            train_loss += loss.item()
            train_total += labels.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
        ep_train_time = time.time() - ep_t0
        train_acc = 100.0 * train_correct / max(train_total, 1)
        if epoch == 0 and diag_n > 0:
            print(f"   🔍 首轮前 {diag_n} batch 耗时分解: 取数+搬运 "
                  f"{diag_fetch / diag_n * 1000:.0f} ms/batch | 前向+反向+更新 "
                  f"{diag_step / diag_n * 1000:.0f} ms/batch"
                  f"（前者大=CPU/内存瓶颈，后者大=GPU算子瓶颈）")

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                _, predicted = torch.max(model(images), 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        val_acc = 100.0 * val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if best_path is not None:
                torch.save(model.state_dict(), best_path)
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"   Epoch [{epoch + 1}/{epochs}] | LR: {current_lr:.2e} "
              f"| Train Loss: {train_loss / len(train_loader):.4f} | Train Acc: {train_acc:.2f}% "
              f"| Val Acc: {val_acc:.2f}% | Best: {best_val_acc:.2f}% | 本轮耗时: {ep_train_time:.0f}s")

    # ===== 用最佳权重在验证集上算最终指标 =====
    if best_path is not None and os.path.exists(best_path):
        model.load_state_dict(torch.load(best_path, map_location=device))
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            probs = F.softmax(model(images), dim=1)
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(probs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    y_prob = np.array(all_probs)

    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    y_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    try:
        auc_macro = roc_auc_score(y_bin, y_prob, average='macro', multi_class='ovr')
    except ValueError:
        auc_macro = float('nan')

    metrics = {
        'seed': seed, 'OA': 100.0 * (y_true == y_pred).mean(),
        'macro_P': p, 'macro_R': r, 'macro_F1': f1, 'macro_AUC': auc_macro,
        'best_path': best_path,
    }
    # 逐类报告 + 混淆矩阵：定位误差来源（如汽油/柴油互混、晚期风化样本被误判）
    target_names = ['Pure Water', 'Gasoline', 'Diesel', 'Mineral Oil']
    print("\n📊 --- 本次运行分类报告 ---")
    print(classification_report(y_true, y_pred, target_names=target_names, digits=4))
    print("混淆矩阵（行=真实, 列=预测）:")
    print(confusion_matrix(y_true, y_pred))
    print(f"   ✅ 完成 | OA={metrics['OA']:.2f}%  macro-F1={f1:.4f}  macro-AUC={auc_macro:.4f}"
          + (f"  (权重: {best_path})" if best_path else ""))
    # 连跑多个实验时主动释放显存，防止前一轮的模型占用累积
    del model, optimizer, scheduler
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return metrics


def _print_mean_std(rows, title):
    """汇总多次运行的 mean ± std（论文 Table II 的呈现格式）。"""
    print(f"\n📊 === {title} ===")
    print(f"{'运行':<14}{'OA(%)':<10}{'macro-F1':<10}{'macro-AUC':<10}")
    for r in rows:
        print(f"seed={r['seed']:<8}{r['OA']:<10.2f}{r['macro_F1']:<10.4f}{r['macro_AUC']:<10.4f}")
    for key in ('OA', 'macro_F1', 'macro_AUC'):
        vals = np.array([r[key] for r in rows], dtype=float)
        print(f"👉 {key}: mean = {vals.mean():.4f} ± {vals.std(ddof=1):.4f} (n={len(rows)})")
    return {key: (float(np.mean([r[key] for r in rows])), float(np.std([r[key] for r in rows], ddof=1)))
            for key in ('OA', 'macro_F1', 'macro_AUC')}


# ==========================================
# 6. 多种子微调（回答"结果稳不稳"）
# ==========================================
def run_multiseed_experiment(weight_path, h5_path, seeds=(42, 43, 44), epochs=50, batch_size=1024,
                             dataset=None):
    """
    同一划分、不同随机种子微调 N 次，产出 N 个独立 pth，报告 mean±std。
    用途：论文主表（Table II）每个模型报告 mean±std，代替单次结果。
    注意：种子覆盖的是初始化/数据打乱的不确定性；划分方式的不确定性用 5 折交叉验证（第7节）。
    dataset 参数可传入已加载的数据集对象，避免连跑时重复加载 HDF5。
    """
    algo = detect_algo(weight_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"multiseed_{algo}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    _reset_logger(os.path.join(out_dir, "multiseed_log.txt"))

    full_dataset = dataset if dataset is not None else OilSpill4ClassDataset(h5_path)
    train_idx, val_idx = stratified_block_split(full_dataset, train_ratio=0.8)

    rows = []
    for seed in seeds:
        rows.append(finetune_once(full_dataset, train_idx, val_idx,
                                  weight_path=weight_path, seed=seed,
                                  epochs=epochs, batch_size=batch_size,
                                  out_dir=out_dir, tag=f"{algo}"))
    stats = _print_mean_std(rows, f"{algo} 多种子结果汇总（{len(seeds)} 个种子）")

    import csv
    with open(os.path.join(out_dir, "multiseed_metrics.csv"), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'OA(%)', 'macro_P', 'macro_R', 'macro_F1', 'macro_AUC', 'best_weight'])
        for r in rows:
            w.writerow([r['seed'], f"{r['OA']:.2f}", f"{r['macro_P']:.4f}", f"{r['macro_R']:.4f}",
                        f"{r['macro_F1']:.4f}", f"{r['macro_AUC']:.4f}", r['best_path']])
    print(f"\n💾 明细已保存: {os.path.join(out_dir, 'multiseed_metrics.csv')}")
    return rows, stats


# ==========================================
# 7. 块级 5 折交叉验证（回答"换个划分还成立吗"）
# ==========================================
def stratified_block_kfold_indices(dataset, k=5):
    """
    类内块级 K 折：每个类别内部按存储顺序（≈源影像×光栅顺序）切成 k 个连续块，
    第 i 折取各类第 i 块作验证、其余 k-1 块作训练。四类比例在每折中天然一致。
    """
    labels = np.asarray(dataset.ram_labels)
    folds = [{'train': [], 'val': []} for _ in range(k)]
    class_names = ['Pure Water', 'Gasoline', 'Diesel', 'Mineral Oil']
    print(f"\n📊 类内块级 {k} 折交叉验证划分：")
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        blocks = np.array_split(idx, k)   # 连续块，保持源影像顺序
        print(f"   -> [{class_names[c]}] 每折验证块约 {len(blocks[0])} 个切片")
        for i in range(k):
            for j in range(k):
                folds[i]['val' if j == i else 'train'].extend(blocks[j].tolist())
    return [(f['train'], f['val']) for f in folds]


def run_block_kfold(weight_path, h5_path, k=5, seed=42, epochs=50, batch_size=1024,
                    dataset=None, algo=None):
    """
    块级 K 折交叉验证：5 折 × 每个模型 = 5 次微调，报告各折指标与 mean±std。
    建议：四个模型跑多种子（第6节）作为主表；最佳模型再跑本函数作为稳健性附录。
      若算力充足，四模型 × 5 折全部跑完是最强证据。
    从头训练版（同协议公平对比）：weight_path=None 且 algo='MAE' 等，即可对
      随机初始化的同架构模型跑同样的 K 折。
    dataset 参数可传入已加载的数据集对象，避免连跑时重复加载 HDF5。
    """
    algo = algo or detect_algo(weight_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"blockCV_{algo}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    _reset_logger(os.path.join(out_dir, "block_cv_log.txt"))

    full_dataset = dataset if dataset is not None else OilSpill4ClassDataset(h5_path)
    folds = stratified_block_kfold_indices(full_dataset, k=k)

    rows = []
    for i, (tr_idx, va_idx) in enumerate(folds):
        print(f"\n===== 折 {i + 1}/{k} =====")
        m = finetune_once(full_dataset, tr_idx, va_idx, weight_path=weight_path,
                          algo=algo if weight_path is None else None,
                          seed=seed, epochs=epochs, batch_size=batch_size,
                          out_dir=out_dir, tag=f"{algo}_fold{i + 1}")
        m['fold'] = i + 1
        rows.append(m)
    stats = _print_mean_std(rows, f"{algo} 块级 {k} 折交叉验证汇总")

    import csv
    with open(os.path.join(out_dir, "block_cv_metrics.csv"), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['fold', 'OA(%)', 'macro_P', 'macro_R', 'macro_F1', 'macro_AUC', 'best_weight'])
        for r in rows:
            w.writerow([r['fold'], f"{r['OA']:.2f}", f"{r['macro_P']:.4f}", f"{r['macro_R']:.4f}",
                        f"{r['macro_F1']:.4f}", f"{r['macro_AUC']:.4f}", r['best_path']])
    print(f"\n💾 明细已保存: {os.path.join(out_dir, 'block_cv_metrics.csv')}")
    return rows, stats


# ==========================================
# 8. 基线实验（定位 SSL 的净收益）
# ==========================================
def _extract_mean_spectra(full_dataset, indices, chunk=20000):
    """提取每个 patch 的 30 维平均光谱（经典机器学习基线的标准特征），分块防内存爆。"""
    feats = np.empty((len(indices), 30), dtype=np.float32)
    for s in range(0, len(indices), chunk):
        e = min(s + chunk, len(indices))
        feats[s:e] = full_dataset.ram_data[np.asarray(indices[s:e])].mean(axis=(2, 3))
    return feats


def run_classical_baselines(h5_path=None, max_train=50000, max_val=20000, seed=42,
                            dataset=None):
    """
    传统机器学习基线：SVM(RBF) 与随机森林，输入为 patch 的 30 维平均光谱。
    与 SSL 模型共用同一套块划分（同 train/val 索引），保证对比公平。
    采样上限默认 5 万/2 万（SVM 复杂度 O(n^2~n^3)，全量 77 万不可行）。
    dataset 参数可传入已加载的数据集对象，避免连跑时重复加载 HDF5。
    """
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    full_dataset = dataset if dataset is not None else OilSpill4ClassDataset(h5_path)
    train_idx, val_idx = stratified_block_split(full_dataset, train_ratio=0.8)

    rng = np.random.RandomState(seed)
    tr_sub = rng.choice(train_idx, size=min(max_train, len(train_idx)), replace=False)
    va_sub = rng.choice(val_idx, size=min(max_val, len(val_idx)), replace=False)
    print(f"\n📊 经典基线采样: 训练 {len(tr_sub)} / 验证 {len(va_sub)}（30维平均光谱特征）")

    Xtr = _extract_mean_spectra(full_dataset, tr_sub)
    ytr = np.asarray(full_dataset.ram_labels)[tr_sub]
    Xva = _extract_mean_spectra(full_dataset, va_sub)
    yva = np.asarray(full_dataset.ram_labels)[va_sub]

    scaler = StandardScaler().fit(Xtr)
    Xtr, Xva = scaler.transform(Xtr), scaler.transform(Xva)

    results = {}
    for name, clf in [('SVM-RBF', SVC(kernel='rbf', C=10, gamma='scale')),
                      ('RandomForest', RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                                              random_state=seed))]:
        print(f"\n🌲 训练 {name} ...")
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xva)
        oa = 100.0 * accuracy_score(yva, pred)
        p, r, f1, _ = precision_recall_fscore_support(yva, pred, average='macro', zero_division=0)
        results[name] = {'OA': oa, 'macro_P': p, 'macro_R': r, 'macro_F1': f1}
        print(f"   ✅ {name}: OA={oa:.2f}%  macro-F1={f1:.4f}")
        print(confusion_matrix(yva, pred))
    return results


def run_linear_probe(weight_path, h5_path=None, epochs=20, batch_size=1024, seed=42,
                     dataset=None):
    """
    Linear Probe：冻结预训练编码器，只训练 Linear(256->4) 线性头。
    这是 SSL 领域的标准评估协议——衡量预训练特征本身的质量（剔除分类头容量的干扰）。
    与全量微调（finetune_once）对比，可分离"特征质量"与"微调策略"各自的贡献。
    dataset 参数可传入已加载的数据集对象，避免连跑时重复加载 HDF5。
    """
    from sklearn.metrics import precision_recall_fscore_support
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(seed)

    full_dataset = dataset if dataset is not None else OilSpill4ClassDataset(h5_path)
    train_idx, val_idx = stratified_block_split(full_dataset, train_ratio=0.8)
    train_loader = DataLoader(Subset(full_dataset, train_idx), batch_size=batch_size,
                              shuffle=True, num_workers=0,
                              pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size,
                            shuffle=False, num_workers=0,
                            pin_memory=torch.cuda.is_available())

    full_model = FineTuneModel(weight_path=weight_path, num_classes=4).to(device)
    encoder = full_model.encoder
    for p in encoder.parameters():
        p.requires_grad = False   # 🔒 冻结编码器
    encoder.eval()

    head = nn.Linear(256, 4).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🔒 Linear Probe: 编码器冻结，仅训练线性头 ({epochs} epochs)")
    for epoch in range(epochs):
        head.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.no_grad():
                feats = encoder(images)
            loss = criterion(head(feats), labels)
            loss.backward()
            optimizer.step()

        head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                pred = head(encoder(images)).argmax(dim=1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"   Epoch [{epoch + 1}/{epochs}] | Val Acc: {100.0 * correct / total:.2f}%")

    all_labels, all_preds = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            all_preds.extend(head(encoder(images)).argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
    y_true, y_pred = np.array(all_labels), np.array(all_preds)
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
    print(f"   ✅ Linear Probe 最终: OA={100.0 * (y_true == y_pred).mean():.2f}%  macro-F1={f1:.4f}")
    return {'OA': 100.0 * (y_true == y_pred).mean(), 'macro_P': p, 'macro_R': r, 'macro_F1': f1}


# ==========================================
# 8.5 标签效率实验（SSL 价值的关键验证：少标签场景）
# ==========================================
def run_label_efficiency_experiment(weight_path, h5_path, ratios=(1.0, 0.25, 0.10, 0.05, 0.01),
                                    seed=42, epochs=50, batch_size=1024, dataset=None):
    """
    标签效率曲线：块级 80/20 划分保持不变，训练侧按类别均匀稀疏抽样到
    100% / 25% / 10% / 5% / 1% 标签，分别微调【预训练】与【同架构从头训练】。

    动机：全量标签(77万)下预训练未必占优（实测甚至略差），但 SSL 的核心价值
    恰恰在标签稀缺场景——标注水面油膜样本代价极高。若少标签时 预训练 > 从头，
    论文核心论点即成立，且这是 SSL 论文的标准评测协议。

    抽样策略：类别均衡 + 时间均匀间隔抽样（覆盖该类 0-48h 全部时段，
    避免只抽到单一老化阶段的偏置）。
    产出：label_efficiency_<ALGO>_<ts>/ 下 label_efficiency.csv + 曲线图(png/pdf)。
    """
    algo = detect_algo(weight_path)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"label_efficiency_{algo}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    _reset_logger(os.path.join(out_dir, "label_efficiency_log.txt"))

    full_dataset = dataset if dataset is not None else OilSpill4ClassDataset(h5_path)
    train_idx, val_idx = stratified_block_split(full_dataset, train_ratio=0.8)
    tr_idx_arr = np.asarray(train_idx)
    tr_labels = np.asarray(full_dataset.ram_labels)[tr_idx_arr]

    print(f"🧪 === {algo} 标签效率实验 | 标签比例: {[f'{r:g}' for r in ratios]} ===")
    rows = []
    for ratio in ratios:
        if ratio >= 1.0:
            sub_idx = list(train_idx)
        else:
            sub_idx = []
            for c in np.unique(tr_labels):
                cls_idx = tr_idx_arr[tr_labels == c]
                k = max(int(round(len(cls_idx) * ratio)), 8)
                pick = np.linspace(0, len(cls_idx) - 1, k).round().astype(int)
                sub_idx.extend(cls_idx[pick].tolist())
            np.random.RandomState(seed).shuffle(sub_idx)
        print(f"\n{'=' * 60}\n📌 标签比例 {ratio * 100:g}% | 训练样本 {len(sub_idx)} / {len(train_idx)}\n{'=' * 60}")

        for mode, wpath in (('pretrained', weight_path), ('scratch', None)):
            r = finetune_once(full_dataset, sub_idx, val_idx,
                              weight_path=wpath, algo=algo if wpath is None else None,
                              seed=seed, epochs=epochs, batch_size=batch_size,
                              out_dir=out_dir, tag=f"{algo}_{mode}_r{ratio:g}")
            r.update({'ratio': ratio, 'mode': mode, 'n_train': len(sub_idx)})
            rows.append(r)

    import csv
    csv_path = os.path.join(out_dir, "label_efficiency.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['ratio', 'mode', 'seed', 'n_train', 'OA(%)', 'macro_F1', 'macro_AUC', 'best_weight'])
        for r in rows:
            w.writerow([r['ratio'], r['mode'], r['seed'], r['n_train'],
                        f"{r['OA']:.2f}", f"{r['macro_F1']:.4f}", f"{r['macro_AUC']:.4f}", r['best_path']])
    print(f"\n💾 标签效率明细已保存: {csv_path}")

    # 汇总打印 + 曲线图（论文可直接用）
    pre = {r['ratio']: r['OA'] for r in rows if r['mode'] == 'pretrained'}
    scr = {r['ratio']: r['OA'] for r in rows if r['mode'] == 'scratch'}
    print(f"\n📊 {algo} 标签效率汇总（OA%）:")
    print(f"{'标签比例':<10}{'预训练':<10}{'从头训练':<10}{'差值':<10}")
    for x in sorted(pre):
        print(f"{x * 100:<9g}%{pre[x]:<10.2f}{scr[x]:<10.2f}{pre[x] - scr[x]:+.2f}")
    xs = sorted(pre)
    plt.figure(figsize=(7, 5), dpi=300)
    plt.plot([x * 100 for x in xs], [pre[x] for x in xs], 'o-', color='#1f77b4',
             linewidth=2.5, markersize=8, label=f'{algo} (pretrained)')
    plt.plot([x * 100 for x in xs], [scr[x] for x in xs], 's--', color='#d62728',
             linewidth=2.5, markersize=8, label=f'{algo} (from scratch)')
    plt.xscale('log')
    plt.xticks([1, 5, 10, 25, 100], ['1%', '5%', '10%', '25%', '100%'])
    plt.xlabel('Training Labels (%)', fontsize=13, fontweight='bold')
    plt.ylabel('Overall Accuracy (%)', fontsize=13, fontweight='bold')
    plt.title(f'Label Efficiency: {algo}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "label_efficiency_curve.png"))
    plt.savefig(os.path.join(out_dir, "label_efficiency_curve.pdf"))
    plt.close()
    print(f"🎨 曲线图已保存: {os.path.join(out_dir, 'label_efficiency_curve.png')}")
    return rows


# ==========================================
# 9. 阈值敏感性分析（0.55 置信门限 / 0.25 NIR 否决）
# ==========================================
def threshold_sensitivity_analysis(model, full_dataset, val_idx, device, batch_size=1024,
                                   conf_grid=None, nir_grid=None, save_dir='.'):
    """
    在验证集上系统性扫描两个推理阈值，回答审稿人"0.55 和 0.25 是怎么来的"：
      1) 置信度门限 conf：max_prob < conf 的 patch 被拒判（计为错误），看 OA-拒判率权衡；
      2) NIR 否决阈值 nir_thr：patch 的 band29 平均反射率 > nir_thr 时被否决（计为错误）。
    NIR 反射率由归一化值逆变换还原：R = x * (MAX - MIN) + MIN（与推理管线同一组绝对极值）。
    ⚠️ 注意：验证集只含已知四类，此分析衡量的是阈值对"已知类样本"的代价；
      阈值对"未知背景"的拦截收益需在实景图上评估（即 Fig.7 的定性证据），两者互补。
    """
    if conf_grid is None:
        conf_grid = np.arange(0.40, 0.75, 0.05)
    if nir_grid is None:
        nir_grid = np.arange(0.15, 0.40, 0.05)

    TRAIN_GLOBAL_MIN, TRAIN_GLOBAL_MAX = -0.2286, 1.1117

    print("\n📏 缓存验证集预测概率与 NIR 反射率（仅前向一次）...")
    model.eval()
    val_loader = DataLoader(Subset(full_dataset, val_idx), batch_size=batch_size,
                            shuffle=False, num_workers=0,
                            pin_memory=torch.cuda.is_available())
    all_probs, all_labels, all_nir = [], [], []
    with torch.no_grad():
        for images, labels in val_loader:
            logits = model(images.to(device))
            all_probs.extend(F.softmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
            # band 29 平均反射率（逆归一化）
            nir_norm = images[:, 29].mean(dim=(1, 2)).numpy()
            all_nir.extend(nir_norm * (TRAIN_GLOBAL_MAX - TRAIN_GLOBAL_MIN) + TRAIN_GLOBAL_MIN)
    probs = np.array(all_probs)
    y_true = np.array(all_labels)
    nir_refl = np.array(all_nir)
    y_pred = probs.argmax(axis=1)
    max_prob = probs.max(axis=1)

    # --- 置信度门限扫描 ---
    print("\n📊 置信度门限敏感性（被拒判样本计为错误）：")
    print(f"{'conf':<8}{'OA(%)':<10}{'拒判率(%)':<12}")
    conf_rows = []
    for conf in conf_grid:
        rejected = max_prob < conf
        oa = 100.0 * ((y_pred == y_true) & ~rejected).mean()
        rej = 100.0 * rejected.mean()
        conf_rows.append((conf, oa, rej))
        print(f"{conf:<8.2f}{oa:<10.2f}{rej:<12.2f}")

    # --- NIR 否决阈值扫描 ---
    print("\n📊 NIR 否决阈值敏感性（被否决样本计为错误）：")
    print(f"{'nir_thr':<8}{'OA(%)':<10}{'否决率(%)':<12}")
    nir_rows = []
    for thr in nir_grid:
        vetoed = nir_refl > thr
        oa = 100.0 * ((y_pred == y_true) & ~vetoed).mean()
        rej = 100.0 * vetoed.mean()
        nir_rows.append((thr, oa, rej))
        print(f"{thr:<8.2f}{oa:<10.2f}{rej:<12.2f}")

    # --- 绘图 ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=300)
    axes[0].plot([r[0] for r in conf_rows], [r[1] for r in conf_rows], 'o-', label='OA')
    axes[0].plot([r[0] for r in conf_rows], [r[2] for r in conf_rows], 's--', label='Rejection rate')
    axes[0].axvline(0.55, color='r', ls=':', lw=1.5, label='Adopted (0.55)')
    axes[0].set_xlabel('Confidence threshold'); axes[0].set_ylabel('%')
    axes[0].set_title('(a) Confidence gate'); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot([r[0] for r in nir_rows], [r[1] for r in nir_rows], 'o-', label='OA')
    axes[1].plot([r[0] for r in nir_rows], [r[2] for r in nir_rows], 's--', label='Veto rate')
    axes[1].axvline(0.25, color='r', ls=':', lw=1.5, label='Adopted (0.25)')
    axes[1].set_xlabel('NIR veto threshold'); axes[1].set_ylabel('%')
    axes[1].set_title('(b) NIR optical veto'); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "07_Threshold_Sensitivity.pdf"))
    plt.savefig(os.path.join(save_dir, "07_Threshold_Sensitivity.png"))
    plt.close()
    print(f"\n🎨 阈值敏感性图已保存: {os.path.join(save_dir, '07_Threshold_Sensitivity.png')}")
    return {'conf': conf_rows, 'nir': nir_rows}


# ==========================================
# 10. 网格级（cell-level）精度评估：让实景解译图从定性变定量
# ==========================================
def cell_level_accuracy(inference_result, cell_id_map=None, cell_class_map=None,
                        min_cell_pixels=500, save_dir='.', grid_layout=None):
    """
    对全图滑窗推理结果做网格级定量评估。10 个围油栏网格的布设类别是已知的，
    等于实景图自带网格级真值——每个网格内多数投票与布设类别比对即可。

    参数:
      inference_result: draw_sliding_window_heatmap 的返回字典（含 pred_map 等）
      cell_id_map:      (H, W) int 数组，0=非网格区域，1..10=网格编号。
                        None 时对非背景区域做连通域自动检测（需人工核对数量与顺序！）
      cell_class_map:   dict {网格编号: 真实类别(0-3)}，0水/1汽油/2柴油/3矿物油
      min_cell_pixels:  自动检测时小于该像素数的连通域视为噪声斑块，忽略
      grid_layout:      (行数, 列数)，如 (5, 2)。给定后跳过自动检测，直接把影像
                        均匀划分成 行×列 网格、行优先编号作为 cell_id_map。
                        格子几何布局规则（浮筒梯子形等）时这是最稳的方案：
                        网格边界落在浮筒/背景黑带上，而背景像素不计入投票。

    输出: 每格的多数投票类别、格内像素级准确率、格级 OA（10 格中判对几格）。
    """
    from scipy import ndimage
    class_names = ['Pure Water', 'Gasoline', 'Diesel', 'Mineral Oil']
    pred_map = inference_result['pred_map']

    if cell_id_map is None and grid_layout is not None:
        rows_g, cols_g = grid_layout
        H, W = pred_map.shape
        cell_id_map = np.zeros_like(pred_map)
        cid = 1
        for r in range(rows_g):
            for c in range(cols_g):
                y0, y1 = H * r // rows_g, H * (r + 1) // rows_g
                x0, x1 = W * c // cols_g, W * (c + 1) // cols_g
                cell_id_map[y0:y1, x0:x1] = cid
                cid += 1
        print(f"📐 按 {rows_g}×{cols_g} 几何网格划分影像（行优先编号，边界落在背景带上）")

    if cell_id_map is None:
        print("⚠️ 未提供 cell_id_map / grid_layout，对非背景区域做连通域自动检测（请人工核对）...")
        fg = (pred_map != 4)
        fg = ndimage.binary_closing(fg, structure=np.ones((3, 3)))
        lab, n = ndimage.label(fg)
        cell_id_map = np.zeros_like(pred_map)
        cid = 1
        for i in range(1, n + 1):
            if (lab == i).sum() >= min_cell_pixels:
                cell_id_map[lab == i] = cid
                cid += 1
        print(f"   -> 自动检测到 {cid - 1} 个候选网格（若数量≠实际格数，建议改用 grid_layout 参数）")

    cell_ids = [i for i in np.unique(cell_id_map) if i != 0]
    if cell_class_map is None:
        print("❌ 必须提供 cell_class_map（每个网格布设的真实类别），例如：")
        print("   cell_class_map = {1: 0, 2: 0, 3: 2, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 3, 10: 3}")
        return None

    print(f"\n📊 === 网格级精度评估（共 {len(cell_ids)} 格）===")
    n_mapped = len(cell_class_map)
    if len(cell_ids) != n_mapped:
        print(f"⚠️ 检测到 {len(cell_ids)} 格，但 cell_class_map 只有 {n_mapped} 格的类别——"
              f"编号对不上！请人工核对热图：多出来的通常是噪声斑块（可调大 min_cell_pixels），"
              f"少掉的通常是相邻格被合并（可调小闭运算核）。未映射的格子将跳过。")
    print(f"{'网格':<6}{'真实类别':<14}{'多数投票':<14}{'格内像素精度':<14}{'判定':<6}")
    rows, correct_cells = [], 0
    for cid in cell_ids:
        mask = cell_id_map == cid
        votes = pred_map[mask]
        votes = votes[votes != 4]          # 剔除背景/未知像素
        if len(votes) == 0:
            print(f"{cid:<6}{'—':<14}{'全部背景':<14}{'—':<14}{'跳过':<6}")
            continue
        maj = int(np.bincount(votes, minlength=4).argmax())
        true_cls = cell_class_map.get(cid)
        if true_cls is None:
            print(f"{cid:<6}{'未映射':<14}{class_names[maj]:<14}{'—':<14}{'跳过':<6}")
            continue
        pix_acc = 100.0 * (votes == true_cls).mean()
        ok = (maj == true_cls)
        correct_cells += ok
        rows.append({'cell': cid, 'true': true_cls, 'voted': maj, 'pix_acc': pix_acc, 'ok': ok})
        print(f"{cid:<6}{class_names[true_cls]:<14}{class_names[maj]:<14}{pix_acc:<14.2f}"
              f"{'✓' if ok else '✗':<6}")

    cell_oa = 100.0 * correct_cells / len(rows) if rows else 0.0
    mean_pix = np.mean([r['pix_acc'] for r in rows]) if rows else 0.0
    print(f"\n👉 网格级 OA: {correct_cells}/{len(rows)} = {cell_oa:.1f}%")
    print(f"👉 格内平均像素精度: {mean_pix:.2f}%")
    print("💡 论文引用建议：'10 个围油栏网格中 X 个被多数投票正确识别，格内平均像素精度 XX.X%'")

    import csv
    csv_path = os.path.join(save_dir, "cell_level_accuracy.csv")
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['cell_id', 'true_class', 'voted_class', 'pixel_acc(%)', 'correct'])
        for r in rows:
            w.writerow([r['cell'], class_names[r['true']], class_names[r['voted']],
                        f"{r['pix_acc']:.2f}", int(r['ok'])])
    print(f"💾 明细已保存: {csv_path}")
    return {'cell_OA': cell_oa, 'mean_pixel_acc': mean_pix, 'rows': rows}


def run_kfold_all(weight_map, h5_path, k=5, seed=42, epochs=50, batch_size=1024):
    """
    一键跑完所有模型的块级 K 折交叉验证（论文主表推荐方案）。

    为什么用它做主表：单次 80/20 尾部块验证 = "最难一块"的严格下界（可能对应晚期
    风化油膜，绝对精度被低估）；而 K 折的验证块覆盖数据序列的不同区段，均值代表
    "平均难度"，既无泄漏又有时间代表性。随机划分的结果含重叠泄漏，不能进论文。
    """
    import csv
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    master_dir = f"kfold_all_{timestamp}"
    os.makedirs(master_dir, exist_ok=True)
    _reset_logger(os.path.join(master_dir, "kfold_all_log.txt"))

    print(f"🧪 === 全部模型 {k} 折交叉验证启动（数据集仅加载一次）===")
    t0 = time.time()
    full_dataset = OilSpill4ClassDataset(h5_path)

    summary = []
    for algo, wpath in weight_map.items():
        try:
            rows, stats = run_block_kfold(wpath, None, k=k, seed=seed, epochs=epochs,
                                          batch_size=batch_size, dataset=full_dataset)
            summary.append((algo, k,
                            stats['OA'][0], stats['OA'][1],
                            stats['macro_F1'][0], stats['macro_F1'][1],
                            stats['macro_AUC'][0], stats['macro_AUC'][1]))
        except Exception as e:
            print(f"❌ {algo} 交叉验证失败（已跳过）: {e}")

    _reset_logger(os.path.join(master_dir, "kfold_all_log.txt"))
    print("\n" + "=" * 78)
    print(f"🏁 全部 {k} 折交叉验证完成！总耗时 {(time.time() - t0) / 3600:.1f} 小时")
    print(f"{'模型':<12}{'折数':<6}{'OA mean±std':<20}{'F1 mean±std':<20}{'AUC mean±std':<20}")
    for name, kk, oa_m, oa_s, f1_m, f1_s, auc_m, auc_s in summary:
        print(f"{name:<12}{kk:<6}{oa_m:>7.2f}±{oa_s:<10.2f}{f1_m:>7.4f}±{f1_s:<10.4f}"
              f"{auc_m:>7.4f}±{auc_s:<10.4f}")
    print("=" * 78)

    with open(os.path.join(master_dir, "kfold_summary.csv"), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['model', 'folds', 'OA_mean', 'OA_std', 'F1_mean', 'F1_std', 'AUC_mean', 'AUC_std'])
        w.writerows(summary)
    print(f"💾 汇总已保存: {os.path.join(master_dir, 'kfold_summary.csv')}")
    return summary


# ==========================================
# 10.5 诊断工具（排查"块划分结果偏低"，结果不进论文）
# ==========================================
def check_block_shift(h5_path=None, dataset=None, head_ratio=0.2, tail_ratio=0.2):
    """
    诊断训练块与验证块之间的光谱漂移：对比每类"序列前 20%"（训练侧）与
    "序列后 20%"（验证侧）的 30 波段平均光谱。

    背景：HDF5 中同类 patch 按采集顺序追加。若实验数据覆盖投放后 0~48h 的风化过程，
    则每类的尾部块很可能对应晚期（高度风化）油膜——块划分无意中变成了时间外推测试：
    验证集是"最难的一块"而非"代表性样本"，表现为绝对精度偏低 + 训练末期验证曲线震荡。

    解读：最大相对差异普遍 >10% 即存在系统性漂移；把生成的 block_shift_diagnostic.png
    发出来即可直观确认。约 2 分钟跑完。
    """
    full_dataset = dataset if dataset is not None else OilSpill4ClassDataset(h5_path)
    labels = np.asarray(full_dataset.ram_labels)
    class_names = ['Pure Water', 'Gasoline', 'Diesel', 'Mineral Oil']
    print("\n🔬 块漂移诊断：各类前 20%（训练侧） vs 后 20%（验证侧）平均光谱")
    print(f"{'类别':<14}{'头部@551nm':<14}{'尾部@551nm':<14}{'最大相对差异':<12}")

    def _mean_spectrum(idxs):
        s = np.zeros(30, dtype=np.float64)
        tot = 0
        for s0 in range(0, len(idxs), 20000):
            chunk = idxs[s0:s0 + 20000]
            arr = np.asarray(full_dataset.ram_data[chunk], dtype=np.float32)  # (n,30,16,16)
            s += arr.mean(axis=(0, 2, 3)) * len(chunk)
            tot += len(chunk)
        return s / tot

    results = {}
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        n_head, n_tail = int(len(idx) * head_ratio), int(len(idx) * tail_ratio)
        spec_head = _mean_spectrum(idx[:n_head])
        spec_tail = _mean_spectrum(idx[-n_tail:])
        rel_diff = np.abs(spec_tail - spec_head) / (np.abs(spec_head) + 1e-8)
        results[class_names[c]] = {'head': spec_head, 'tail': spec_tail, 'rel_diff': rel_diff}
        print(f"{class_names[c]:<14}{spec_head[7]:<14.4f}{spec_tail[7]:<14.4f}{rel_diff.max():<12.1%}")

    print("\n💡 解读：最大相对差异普遍 >10% = 验证侧光谱系统性漂移（时间风化/光照差异），")
    print("   块划分的绝对精度被低估、曲线震荡属预期；5 折交叉验证的均值可代表'平均难度'。")

    try:
        wavelengths = np.linspace(409, 990, 30)
        plt.figure(figsize=(9, 5.5), dpi=200)
        for cname, d in results.items():
            plt.plot(wavelengths, d['head'], '-',  label=f'{cname} head (train side)')
            plt.plot(wavelengths, d['tail'], '--', label=f'{cname} tail (val side)')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Mean (dataset-normalized units)')
        plt.title('Head-block vs tail-block mean spectra per class')
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig('block_shift_diagnostic.png')
        plt.close()
        print("🎨 光谱对比图已保存: block_shift_diagnostic.png")
    except Exception as e:
        print(f"（绘图跳过: {e}）")
    return results


def random_split_diagnostic(weight_path, h5_path=None, seed=42, epochs=50, batch_size=1024,
                            dataset=None):
    """
    【诊断专用，结果严禁写进论文】随机 80/20 划分（含空间重叠泄漏）跑一次微调，
    用于分解精度损失来源：
      随机划分精度 - 块划分精度 ≈ 重叠泄漏贡献的虚高部分；
      若随机划分同样上不去（<75%），说明数据自身的时间/空间漂移才是主因，
      应优先重建 HDF5 并按影像（而非按序列尾部）划分。
    """
    full_dataset = dataset if dataset is not None else OilSpill4ClassDataset(h5_path)
    set_seed(seed)
    n = len(full_dataset)
    perm = np.random.permutation(n)
    n_train = int(n * 0.8)
    out_dir = f"random_split_diag_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(out_dir, exist_ok=True)
    _reset_logger(os.path.join(out_dir, "random_diag_log.txt"))
    print("⚠️ 随机划分诊断运行：此结果含空间重叠泄漏，仅用于对照分析，不要写进论文！")
    return finetune_once(full_dataset, perm[:n_train].tolist(), perm[n_train:].tolist(),
                         weight_path=weight_path, seed=seed, epochs=epochs,
                         batch_size=batch_size, out_dir=out_dir, tag='random_diag')


# ==========================================
# 11. 从头训练基线（多种子包装）与一键总控
# ==========================================
def run_scratch_baseline(algo, h5_path=None, seeds=(42, 43, 44), epochs=50, batch_size=1024,
                         dataset=None, encoder_lr=1e-4):
    """
    从头训练基线的多种子版：同架构、同划分、同协议，但不加载任何预训练权重。
    与 run_multiseed_experiment 的预训练结果直接对比，差值即 SSL 预训练的净收益。
    algo ∈ {'MAE', 'DAE', 'SIMCLR', 'SIMSIAM'}；四种架构各调一次。
    """
    algo = algo.upper()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = f"scratch_{algo}_{timestamp}"
    os.makedirs(out_dir, exist_ok=True)
    _reset_logger(os.path.join(out_dir, "scratch_log.txt"))

    full_dataset = dataset if dataset is not None else OilSpill4ClassDataset(h5_path)
    train_idx, val_idx = stratified_block_split(full_dataset, train_ratio=0.8)

    rows = []
    for seed in seeds:
        rows.append(finetune_once(full_dataset, train_idx, val_idx,
                                  weight_path=None, algo=algo, seed=seed,
                                  epochs=epochs, batch_size=batch_size,
                                  out_dir=out_dir, tag=f"{algo}_scratch",
                                  encoder_lr=encoder_lr))
    stats = _print_mean_std(rows, f"{algo} 从头训练基线汇总（{len(seeds)} 个种子）")

    import csv
    with open(os.path.join(out_dir, "scratch_metrics.csv"), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['seed', 'OA(%)', 'macro_P', 'macro_R', 'macro_F1', 'macro_AUC', 'best_weight'])
        for r in rows:
            w.writerow([r['seed'], f"{r['OA']:.2f}", f"{r['macro_P']:.4f}", f"{r['macro_R']:.4f}",
                        f"{r['macro_F1']:.4f}", f"{r['macro_AUC']:.4f}", r['best_path']])
    return rows, stats


def run_all_experiments(weight_map, h5_path, seeds=(42, 43, 44), kfold_model=None,
                        epochs=50, batch_size=1024):
    """
    一键跑完全部投稿补充实验（建议夜间运行）。数据集只加载一次，各阶段共享。

    参数:
      weight_map:   {'MAE': '完整权重路径', 'DAE': ..., 'SIMCLR': ..., 'SIMSIAM': ...}
      seeds:        多种子实验的种子列表（默认 3 个）
      kfold_model:  需要做块级5折交叉验证的模型名（如 'MAE'）；None 跳过
      epochs:       每次微调的轮数（默认 50，与论文协议一致）

    流程与产出（每阶段独立目录 + CSV，单个失败不影响其他阶段，中断不丢已完成结果）:
      1) 4 模型 × 3 种子预训练微调   -> multiseed_*/multiseed_metrics.csv
      2) 4 架构 × 3 种子从头训练基线 -> scratch_*/scratch_metrics.csv
      3) SVM / 随机森林经典基线      -> 直接打印（共用同一划分）
      4) 4 模型 Linear Probe         -> 直接打印
      5) (可选) 最佳模型块级 5 折    -> blockCV_*/block_cv_metrics.csv
      6) 总汇总表                    -> all_experiments_*/summary.csv
    """
    import csv
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    master_dir = f"all_experiments_{timestamp}"
    os.makedirs(master_dir, exist_ok=True)
    _reset_logger(os.path.join(master_dir, "master_log.txt"))

    print("🧪 === 一键全量实验启动 ===")
    print(f"   模型: {list(weight_map.keys())} | 种子: {seeds} | 5折模型: {kfold_model or '跳过'}")
    t0 = time.time()

    print("\n📦 数据集加载（全程仅此一次）...")
    full_dataset = OilSpill4ClassDataset(h5_path)

    summary = []  # 每行: (实验名, OA_mean, OA_std, F1_mean, F1_std, AUC_mean, AUC_std)

    def _add_summary(name, stats):
        summary.append((name,
                        stats['OA'][0], stats['OA'][1],
                        stats['macro_F1'][0], stats['macro_F1'][1],
                        stats['macro_AUC'][0], stats['macro_AUC'][1]))

    # ---- 阶段 1/2：逐模型多种子（预训练 + 从头基线） ----
    for algo, wpath in weight_map.items():
        try:
            _, stats = run_multiseed_experiment(wpath, None, seeds=seeds, epochs=epochs,
                                                batch_size=batch_size, dataset=full_dataset)
            _add_summary(f"{algo} (pretrained)", stats)
        except Exception as e:
            print(f"❌ {algo} 多种子实验失败（已跳过，不影响后续）: {e}")
        try:
            _, stats = run_scratch_baseline(algo, seeds=seeds, epochs=epochs,
                                            batch_size=batch_size, dataset=full_dataset)
            _add_summary(f"{algo} (from scratch)", stats)
        except Exception as e:
            print(f"❌ {algo} 从头基线失败（已跳过，不影响后续）: {e}")

    # ---- 阶段 3：经典机器学习基线 ----
    try:
        cls_results = run_classical_baselines(dataset=full_dataset)
        for name, r in cls_results.items():
            summary.append((name, r['OA'], 0.0, r['macro_F1'], 0.0, float('nan'), 0.0))
    except Exception as e:
        print(f"❌ 经典基线失败（已跳过）: {e}")

    # ---- 阶段 4：Linear Probe（冻结编码器 + 线性头） ----
    for algo, wpath in weight_map.items():
        try:
            r = run_linear_probe(wpath, dataset=full_dataset)
            summary.append((f"{algo} (linear probe)", r['OA'], 0.0, r['macro_F1'], 0.0,
                            float('nan'), 0.0))
        except Exception as e:
            print(f"❌ {algo} Linear Probe 失败（已跳过）: {e}")

    # ---- 阶段 5：块级 5 折交叉验证（可选） ----
    if kfold_model and kfold_model in weight_map:
        try:
            _, stats = run_block_kfold(weight_map[kfold_model], None, k=5, epochs=epochs,
                                       batch_size=batch_size, dataset=full_dataset)
            _add_summary(f"{kfold_model} (5-fold CV)", stats)
        except Exception as e:
            print(f"❌ 块级交叉验证失败（已跳过）: {e}")

    # ---- 阶段 6：总汇总 ----
    _reset_logger(os.path.join(master_dir, "master_log.txt"))
    print("\n" + "=" * 78)
    print("🏁 全部实验完成！总耗时 {:.1f} 小时".format((time.time() - t0) / 3600))
    print(f"{'实验':<26}{'OA mean±std':<18}{'F1 mean±std':<18}{'AUC mean±std':<18}")
    for row in summary:
        name, oa_m, oa_s, f1_m, f1_s, auc_m, auc_s = row
        auc_str = f"{auc_m:.4f}±{auc_s:.4f}" if not np.isnan(auc_m) else "—"
        print(f"{name:<26}{oa_m:>7.2f}±{oa_s:<8.2f}{f1_m:>7.4f}±{f1_s:<8.4f}{auc_str:<18}")
    print("=" * 78)

    with open(os.path.join(master_dir, "summary.csv"), 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['experiment', 'OA_mean', 'OA_std', 'F1_mean', 'F1_std', 'AUC_mean', 'AUC_std'])
        w.writerows(summary)
    print(f"💾 总汇总已保存: {os.path.join(master_dir, 'summary.csv')}")
    return summary


if __name__ == "__main__":
    # 1. 你制作好的用于训练的 H5 数据库
    H5_DATABASE_PATH = r"E:\Dsw\Dataset\OilSpill_Dataset.h5"

    # 2. 四个模型的预训练权重路径（"all" 一键模式 与 "figures" 图件模式共用）
    #    ⚠️ MAE 请用完整权重 mae_pretrained_full.pth（含 patch_embed/cls_token/pos_embed，可完整加载；
    #       旧的 mae_pretrained_encoder.pth 只能加载 Transformer 层，位置编码会缺失）
    WEIGHT_MAP = {
        'MAE':     r"weights/mae/mae_pretrained_full.pth",
        'DAE':     r"weights/dae/dae_pretrained.pth",
        'SIMCLR':  r"weights/simclr/simclr_pretrained.pth",
        'SIMSIAM': r"weights/simsiam/simsiam_pretrained.pth",
    }

    # 3. 你的 10 格子真实测试影像的 .hdr 路径（"single" / "figures" 模式需要）
    TEST_IMAGE_HDR = r"E:\Dsw\Dataset\0414111801.hdr"

    # 4. ★ 运行模式（只改这一个字符串）★
    #    "all"     —— 一键全量实验：4模型×3种子微调 + 4架构×3种子从头基线 + SVM/RF
    #                + 4模型Linear Probe + KFOLD_MODEL 的5折交叉验证 + 总汇总表
    #                （只出数字，不出图；建议整夜运行）
    #    "kfold"   —— 只对 4 个模型各跑块级 5 折 CV（共 20 次微调，≈7 小时），
    #                产出 kfold_all_*/kfold_summary.csv —— 论文 Table II 的主数据来源
    #    "labels"  —— 标签效率实验：MAE 预训练 vs 从头训练，100%/25%/10%/5%/1% 标签
    #                （共 10 次微调；低比例档很快，整夜轻松跑完）
    #                —— SSL 价值的关键验证，少标签占优则论文核心论点成立
    #    "figures" —— 四模型依次训练(seed 42) 并各出一套 6 张论文图
    #                （修复权重加载后，旧图全部作废，需要重新出图）
    #    "eval"    —— 不训练，跳过分支，直接执行下方取消注释的分析段
    #                （阈值敏感性 / 网格级精度；需已存在训练好的权重 pth）
    #    "single"  —— 只跑 SINGLE_MODEL 指定的一个模型：训练 + 出图
    MODE = "eval"
    SINGLE_MODEL = 'MAE'     # "single" 模式下用哪个模型
    KFOLD_MODEL = 'MAE'      # "all" 模式下对哪个模型做 5 折（跑完看结果改成最佳模型再跑）

    if MODE == "all":
        run_all_experiments(WEIGHT_MAP, H5_DATABASE_PATH, seeds=(42, 43, 44), kfold_model=KFOLD_MODEL)
    elif MODE == "kfold":
        run_kfold_all(WEIGHT_MAP, H5_DATABASE_PATH, k=5)
    elif MODE == "labels":
        run_label_efficiency_experiment(WEIGHT_MAP['MAE'], H5_DATABASE_PATH)
    elif MODE == "figures":
        for algo, wpath in WEIGHT_MAP.items():
            print(f"\n{'=' * 60}\n📷 开始为 {algo} 生成论文图表...\n{'=' * 60}")
            run_pipeline(wpath, H5_DATABASE_PATH, TEST_IMAGE_HDR, epochs=50, batch_size=1024)
    elif MODE == "eval":
        print("🔎 eval 模式：跳过训练，直接执行下方分析段（阈值敏感性 / 网格级精度）")
    else:  # "single"
        run_pipeline(WEIGHT_MAP[SINGLE_MODEL], H5_DATABASE_PATH, TEST_IMAGE_HDR, epochs=50, batch_size=1024)

    # 【论文主表】四模型块级5折交叉验证一键连跑（推荐替代单次尾部块验证）：
    # run_kfold_all(WEIGHT_MAP, H5_DATABASE_PATH, k=5)

    # 【诊断】块划分结果偏低时的排查工具（结果不进论文）：
    # check_block_shift(H5_DATABASE_PATH)                              # 2分钟：训练/验证块光谱漂移检查
    # random_split_diagnostic(WEIGHT_MAP['MAE'], H5_DATABASE_PATH)     # 2小时：随机划分对照，量化泄漏贡献

    # 【单项入口】（连跑时建议都传 dataset= 已加载的数据集，避免重复加载 HDF5）
    # run_multiseed_experiment(WEIGHT_MAP['MAE'], H5_DATABASE_PATH, seeds=(42, 43, 44))  # 多种子
    # run_block_kfold(WEIGHT_MAP['MAE'], H5_DATABASE_PATH, k=5)                          # 块级5折
    # run_block_kfold(None, H5_DATABASE_PATH, k=5, algo='MAE')                            # 从头训练的5折（同协议公平对比）
    # run_classical_baselines(H5_DATABASE_PATH)                                       # SVM+RF
    # run_scratch_baseline('MAE', H5_DATABASE_PATH, seeds=(42, 43, 44))               # 从头基线
    # run_linear_probe(WEIGHT_MAP['MAE'], H5_DATABASE_PATH)                              # Linear Probe

    # 【阈值敏感性】扫描 0.55 置信门限与 0.25 NIR 否决（需先训练/加载好模型）
    # ⚠️ 权重文件在哪：figures/single 模式会生成 results_MAE_<时间戳>\MAE_best_finetuned.pth，
    #    把下面 r"E:\Dsw\xunliandaima\results_MAE_20261015_171205\MAE_best_finetuned.pth" 替换成这个实际路径即可（两段共用同一个 model）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_ds = OilSpill4ClassDataset(H5_DATABASE_PATH)
    _, va_idx = stratified_block_split(full_ds, 0.8)
    model = FineTuneModel(weight_path=WEIGHT_MAP['MAE']).to(device)
    model.load_state_dict(torch.load(r"E:\Dsw\xunliandaima\results_MAE_20261015_171205\MAE_best_finetuned.pth", map_location=device))
    threshold_sensitivity_analysis(model, full_ds, va_idx, device, save_dir='.')

    # 【网格级精度】对滑窗推理结果做 10 格定量评估
    result = draw_sliding_window_heatmap(model, device, TEST_IMAGE_HDR, save_dir='.')
    cell_class_map = {1: 0, 2: 0, 3: 2, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 3, 10: 3}
    # ↑ 已按实际布设确认：左上→右下格子1~10 = 水,水,柴油,汽油,汽油,汽油,柴油,柴油,矿物油,矿物油
    cell_level_accuracy(result, cell_id_map=None, cell_class_map=cell_class_map,
                         grid_layout=(5, 2))   # 2列×5行浮筒布局，行优先编号=左上→右下1~10
