import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE

# 导入你原版的网络架构和数据集读取器
from Evaluate_FineTuning import FineTuneModel
from dataset_utils import OilSpill4ClassDataset

# ==========================================
# 0. 全局字体与 SCI 格式设置
# ==========================================
plt.rcParams.update({
    "font.family": 'serif',
    "font.serif": ['Times New Roman'],
    "font.size": 14,
    "axes.unicode_minus": False
})


def get_macro_roc(model, val_loader, device):
    """计算单个模型的宏平均 ROC 数据"""
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    y_test_bin = label_binarize(all_labels, classes=[0, 1, 2, 3])
    n_classes = y_test_bin.shape[1]

    fpr, tpr = dict(), dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], all_probs[:, i])

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= n_classes

    macro_auc = auc(all_fpr, mean_tpr)
    return all_fpr, mean_tpr, macro_auc


def draw_combined_roc(model_dict, h5_path, save_name="Combined_ROC_Curves"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n📐 开始生成多模型联合 ROC 曲线...")

    # 严格还原验证集划分
    full_dataset = OilSpill4ClassDataset(h5_path)
    torch.manual_seed(42)
    total_size = len(full_dataset)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size])
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)

    plt.figure(figsize=(9, 8), dpi=300)
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guessing')

    # 针对不同算法设计不同的颜色和线型，突出对比
    style_map = {
        'MAE': {'color': '#d62728', 'ls': '-', 'lw': 3.5, 'zorder': 1},  # 红色，粗实线 (主角)
        'DAE': {'color': '#ff7f0e', 'ls': '-', 'lw': 2.5, 'zorder': 2},  # 橙色，实线
        'SimCLR': {'color': '#1f77b4', 'ls': '--', 'lw': 2.5, 'zorder': 3},  # 蓝色，虚线
        'SimSiam': {'color': '#2ca02c', 'ls': ':', 'lw': 2.5, 'zorder': 4}  # 绿色，点线
    }

    for algo_name, weight_path in model_dict.items():
        print(f"⏳ 正在计算 {algo_name} 的推理概率...")
        # 这里的 base_weight 只是为了实例化网络结构，载入具体的权重在 load_state_dict 中进行
        # 假设所有算法的骨干网络都是 SimpleEncoder
        base_weight = r"weights/mae/mae_pretrained_encoder.pth"
        model = FineTuneModel(base_weight, num_classes=4).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()

        fpr, tpr, macro_auc = get_macro_roc(model, val_loader, device)

        sty = style_map.get(algo_name, {'color': 'black', 'ls': '-', 'lw': 2, 'zorder': 1})
        plt.plot(fpr, tpr, color=sty['color'], linestyle=sty['ls'], linewidth=sty['lw'], zorder=sty['zorder'],
                 label=f'{algo_name} (Macro AUC = {macro_auc:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=15, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=15, fontweight='bold')
    plt.title('Macro-average ROC Comparison of SSL Models', fontsize=17, fontweight='bold', pad=15)
    plt.legend(loc="lower right", fontsize=14, frameon=True, edgecolor='black')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(f"{save_name}.pdf", bbox_inches='tight')
    plt.savefig(f"{save_name}.png", bbox_inches='tight')
    print(f"✅ 联合 ROC 曲线已保存！")


def draw_combined_tsne(model_dict, h5_path, save_name="Combined_tSNE"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\n🧠 开始生成多模型联合 t-SNE 2x2 矩阵图 (可能需要几分钟)...")

    full_dataset = OilSpill4ClassDataset(h5_path)
    np.random.seed(42)
    subset_indices = np.random.choice(len(full_dataset), 4000, replace=False)
    dataset = Subset(full_dataset, subset_indices)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=300)
    axes = axes.flatten()

    class_names = ['Pure Water', 'Gasoline', 'Diesel', 'Mineral Oil']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for idx, (algo_name, weight_path) in enumerate(model_dict.items()):
        if idx >= 4: break  # 最多画4个
        print(f"⏳ 正在计算 {algo_name} 的高维特征与降维...")

        base_weight = r"weights/mae/mae_pretrained_encoder.pth"
        model = FineTuneModel(base_weight, num_classes=4).to(device)
        model.load_state_dict(torch.load(weight_path, map_location=device))
        model.eval()

        all_features, all_labels = [], []
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                features = model.encoder(images)
                all_features.append(features.cpu().numpy())
                all_labels.append(labels.numpy())

        all_features = np.concatenate(all_features, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        tsne = TSNE(n_components=2, perplexity=40, random_state=42)
        tsne_results = tsne.fit_transform(all_features)

        ax = axes[idx]
        for i, name in enumerate(class_names):
            c_idx = (all_labels == i)
            ax.scatter(tsne_results[c_idx, 0], tsne_results[c_idx, 1],
                       c=colors[i], label=name, alpha=0.7, s=10, edgecolors='none')

        ax.set_title(f"({chr(97 + idx)}) {algo_name} Feature Space", fontsize=16, fontweight='bold', pad=10)
        ax.set_xticks([])
        ax.set_yticks([])

        # 在底部居中放置共享的图例
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=14, bbox_to_anchor=(0.5, -0.02), markerscale=3)

    plt.tight_layout()
    plt.savefig(f"{save_name}.pdf", bbox_inches='tight')
    plt.savefig(f"{save_name}.png", bbox_inches='tight')
    print(f"✅ 联合 t-SNE 2x2 图已保存！")


if __name__ == "__main__":
    # ==========================================
    # ⚙️ 准备工作：填入你跑完 4 个算法后的最优权重路径
    # ==========================================
    H5_DATABASE_PATH = r"E:\Dsw\Dataset\OilSpill_Dataset.h5"

    # 请把这里等号右边的路径，替换成你对应的真实文件路径
    # 字典的 Key 就是画在图里的标签名 (建议按性能从低到高或对比分组排序)
    MODELS_TO_COMPARE = {
        'SimSiam': r"E:\Dsw\xunliandaima\results_SIMSIAM_20261010_220454\SIMSIAM_best_finetuned.pth",
        'SimCLR': r"E:\Dsw\xunliandaima\results_SIMCLR_20261010_214606\SIMCLR_best_finetuned.pth",
        'DAE': r"E:\Dsw\xunliandaima\results_DAE_20261010_212440\DAE_best_finetuned.pth",
        'MAE': r"E:\Dsw\xunliandaima\results_MAE_20261010_210305\MAE_best_finetuned.pth"
    }

    # 执行联合绘制
    draw_combined_roc(MODELS_TO_COMPARE, H5_DATABASE_PATH)
    draw_combined_tsne(MODELS_TO_COMPARE, H5_DATABASE_PATH)