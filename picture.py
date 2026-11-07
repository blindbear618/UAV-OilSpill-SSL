import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np

# 设置全局字体为 Times New Roman（IEEE 期刊标准）
plt.rcParams['font.sans-serif'] = 'Times New Roman'
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.linewidth'] = 0.8

# 1. 读取三张原图（请确保文件名与本地一致）
img_a = Image.open(r"D:\Doc\博士毕业课题\IMG_20260411_153648.jpg")  # 无人机+水池+定标板全景
img_b = Image.open(r"D:\Doc\博士毕业课题\IMG_20260409_110748.jpg")  # 实验人员现场布设近景
img_c = Image.open(r"D:\Doc\博士毕业课题\339a2dde-04d8-48f3-854f-91ab17196450.jpg")         # 无人机正射俯视图

# ==========================================
# 生成【方案 B：左二右一 紧凑高清组合图】（强烈推荐，不损失横竖画幅细节）
# ==========================================
fig = plt.figure(figsize=(10, 6.5), dpi=300)
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[1, 1], wspace=0.06, hspace=0.08)

# --- 子图 (a)：左上（适度裁剪顶部多余天空与底部多余砖地，突出无人机、水池和定标板）---
ax_a = fig.add_subplot(gs[0, 0])
w_a, h_a = img_a.size
crop_a = img_a.crop((0, int(h_a * 0.18), w_a, int(h_a * 0.90)))
ax_a.imshow(crop_a)
ax_a.axis('off')

# 在图(a)中添加无人机和定标板的标注框與文字
cw_a, ch_a = crop_a.size
# 无人机标注框
rect_uav = patches.Rectangle((cw_a*0.48, ch_a*0.05), cw_a*0.10, ch_a*0.12,
                             linewidth=1.5, edgecolor='#FFD700', facecolor='none', linestyle='-')
ax_a.add_patch(rect_uav)
ax_a.text(cw_a*0.59, ch_a*0.12, 'UAV Platform', color='#FFD700', fontsize=10, fontweight='bold',
          bbox=dict(facecolor='black', alpha=0.5, pad=1.5, edgecolor='none'))

# 辐射定标板标注框
rect_panel = patches.Rectangle((cw_a*0.42, ch_a*0.85), cw_a*0.14, ch_a*0.11,
                               linewidth=1.5, edgecolor='#00FFCC', facecolor='none', linestyle='-')
ax_a.add_patch(rect_panel)
ax_a.text(cw_a*0.58, ch_a*0.92, 'Calibration Panel', color='#00FFCC', fontsize=10, fontweight='bold',
          bbox=dict(facecolor='black', alpha=0.5, pad=1.5, edgecolor='none'))

# 子图(a)角标
ax_a.text(0.02, 0.94, '(a)', transform=ax_a.transAxes, fontsize=13, fontweight='bold',
          va='top', ha='left', color='white', bbox=dict(facecolor='black', alpha=0.6, pad=2, edgecolor='none'))

# --- 子图 (b)：左下（现场油膜布设与操作近景）---
ax_b = fig.add_subplot(gs[1, 0])
ax_b.imshow(img_b)
ax_b.axis('off')
ax_b.text(0.02, 0.94, '(b)', transform=ax_b.transAxes, fontsize=13, fontweight='bold',
          va='top', ha='left', color='white', bbox=dict(facecolor='black', alpha=0.6, pad=2, edgecolor='none'))

# --- 子图 (c)：右侧通栏（无人机正射10围隔俯视图）---
ax_c = fig.add_subplot(gs[:, 1])
ax_c.imshow(img_c)
ax_c.axis('off')
ax_c.text(0.03, 0.97, '(c)', transform=ax_c.transAxes, fontsize=13, fontweight='bold',
          va='top', ha='left', color='white', bbox=dict(facecolor='black', alpha=0.6, pad=2, edgecolor='none'))

# 可选：在图(c)的10个围隔中标注围隔编号与油品类别（对应论文2x5布局）
# 左列 (1, 3, 5, 7, 9) 与 右列 (2, 4, 6, 8, 10) 的大致相对坐标
enclosures_info = [
    (0.31, 0.73, "1: Pure Water"),   (0.57, 0.73, "2: Pure Water"),
    (0.31, 0.62, "3: Diesel"),       (0.57, 0.62, "4: Gasoline"),
    (0.31, 0.51, "5: Gasoline"),     (0.57, 0.51, "6: Gasoline"),
    (0.31, 0.39, "7: Diesel"),       (0.57, 0.39, "8: Diesel"),
    (0.31, 0.26, "9: Mineral Oil"),  (0.57, 0.26, "10: Mineral Oil")
]
for x_rel, y_rel, label in enclosures_info:
    ax_c.text(x_rel, y_rel, label, transform=ax_c.transAxes, fontsize=8.5, fontweight='bold',
              color='white', ha='center', va='center',
              bbox=dict(facecolor='black', alpha=0.45, pad=1.2, edgecolor='none', boxstyle='round,pad=0.2'))

plt.savefig("Fig1_Experimental_Setup_TGRS.png", dpi=300, bbox_inches='tight', pad_inches=0.05)
plt.savefig("Fig1_Experimental_Setup_TGRS.pdf", dpi=300, bbox_inches='tight', pad_inches=0.05)
print("已成功生成高清论文插图：Fig1_Experimental_Setup_TGRS.png / .pdf")