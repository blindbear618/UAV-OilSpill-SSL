import numpy as np
import spectral.io.envi as envi
from scipy.signal import savgol_filter
import scipy.ndimage as ndi
from skimage import measure
from pathlib import Path
from tqdm import tqdm


# ==========================================
# 1. 物理清洗与网格提取模块 (保持核心逻辑不变)
# ==========================================
def remove_sun_glint(reflectance_cube, nir_band_idx=29, target_nir=0.01):
    glint_corrected = np.copy(reflectance_cube)
    nir_band = reflectance_cube[:, :, nir_band_idx]
    delta_glint = nir_band - target_nir
    delta_glint[delta_glint < 0] = 0
    for i in range(reflectance_cube.shape[2]):
        glint_corrected[:, :, i] -= delta_glint
    return np.clip(glint_corrected, 0, 1)


def auto_segment_orange_grid(data_cube, red_idx=12, green_idx=7, min_area=100):
    red_band = data_cube[:, :, red_idx]
    green_band = data_cube[:, :, green_idx]

    orange_ratio = red_band / (green_band + 1e-5)
    mean_red = np.mean(red_band)

    grid_mask = (orange_ratio > 0.95) & (red_band > mean_red * 1.5)

    bw = 5
    grid_mask[:bw, :] = True
    grid_mask[-bw:, :] = True
    grid_mask[:, :bw] = True
    grid_mask[:, -bw:] = True

    struct = ndi.generate_binary_structure(2, 2)
    grid_mask = ndi.binary_dilation(grid_mask, structure=struct, iterations=5)

    water_mask = ~grid_mask
    labeled_mask = measure.label(water_mask, connectivity=1)
    props = measure.regionprops(labeled_mask)

    valid_pools = []
    for prop in props:
        if prop.area > min_area:
            valid_pools.append({
                'label_id': prop.label,
                'centroid': prop.centroid,
                'bbox': prop.bbox,
                'area': prop.area
            })

    # 如果检测到超过 10 个水池，按面积从大到小排序，只取前 10 个最大的
    if len(valid_pools) > 10:
        valid_pools.sort(key=lambda p: p['area'], reverse=True)
        valid_pools = valid_pools[:10]
        # print("  -> 已自动剔除边缘碎片，保留最大的 10 个水池。")

    # 几何排序逻辑 (保持不变，先按 Y 排序，再按 X 排序)
    valid_pools.sort(key=lambda p: p['centroid'][0])
    ordered_pools = []
    for row_idx in range(len(valid_pools) // 2):
        row_pools = valid_pools[row_idx * 2: (row_idx + 1) * 2]
        row_pools.sort(key=lambda p: p['centroid'][1])
        ordered_pools.extend(row_pools)

    return ordered_pools, labeled_mask


# ==========================================
# 2. 导出小水池预览流水线
# ==========================================
def export_pool_previews(input_dir, preview_output_dir):
    # 物理排列标签
    PHYSICAL_LABELS = [
        "00_PureWater1", "01_PureWater2",
        "02_Diesel0_10um", "03_Gasoline92_10um",
        "04_Gasoline92_50m", "05_Gasoline92_2000um",
        "06_Diesel0_50um", "07_Diesel0_2000um",
        "08_MineralOil_50um", "09_MineralOil_2000um"
    ]

    # 为每个油种创建独立的输出文件夹
    for label in PHYSICAL_LABELS:
        (Path(preview_output_dir) / label).mkdir(parents=True, exist_ok=True)

    spe_files = list(Path(input_dir).rglob("*.dat"))
    print(f"共找到 {len(spe_files)} 张包含网格的 .dat 图像影像。")

    for spe_path in tqdm(spe_files, desc="裁剪与导出预览"):
        hdr_path = spe_path.with_suffix('.hdr')
        img = envi.open(hdr_path, spe_path)
        roi = np.array(img.load(), dtype=np.float32)

        # 180度图像自动纠正
        if "_180" in spe_path.name:
            roi = np.rot90(roi, k=2, axes=(0, 1)).copy()

        # 执行网格提取
        ordered_pools, label_matrix = auto_segment_orange_grid(roi, red_idx=12, green_idx=7)

        if len(ordered_pools) != 10:
            print(f"\n跳过 {spe_path.name}: 提取到 {len(ordered_pools)} 个水池。")
            continue

        base_name = spe_path.stem.replace("_180", "")

        # 提取真彩色合成所需的波段 (请根据你的相机波段确认)
        for i, pool in enumerate(ordered_pools):
            pool_name = PHYSICAL_LABELS[i]
            y1, x1, y2, x2 = pool['bbox']
            local_roi = roi[y1:y2, x1:x2, :]
            exact_pool_mask = (label_matrix[y1:y2, x1:x2] == pool['label_id']).astype(np.float32)

            # --- 1. 导出完整的 30 波段数据 ---
            meta_30b = img.metadata.copy()
            meta_30b.pop('lines', None)
            meta_30b.pop('samples', None)
            meta_30b.pop('bands', None)

            out_path_30b_hdr = Path(preview_output_dir) / pool_name / f"{base_name}_30_FullBands.hdr"
            envi.save_image(str(out_path_30b_hdr), local_roi, metadata=meta_30b, dtype=np.float32, interleave='bsq',
                            force=True, ext='.dat')

            # --- 2. 导出单波段掩码文件 ---
            meta_mask = img.metadata.copy()
            meta_mask.pop('lines', None)
            meta_mask.pop('samples', None)
            meta_mask.pop('bands', None)
            # 💡 极其重要：掩码只有 1 个波段，必须从元数据中抹除 30 波长的列表！否则 ENVI 会再次报错
            meta_mask.pop('wavelength', None)
            meta_mask.pop('fwhm', None)

            mask_img = np.expand_dims(exact_pool_mask, axis=2)
            out_path_mask_hdr = Path(preview_output_dir) / pool_name / f"{base_name}_30_Mask.hdr"
            envi.save_image(str(out_path_mask_hdr), mask_img, metadata=meta_mask, dtype=np.float32, interleave='bsq',
                            force=True, ext='.dat')

if __name__ == "__main__":
    # 1. 你在 ENVI 里粗略裁好的网格影像所在文件夹
    INPUT_DIR = r"D:\Doc\博士毕业课题\裁剪_30m"

    # 2. 专门存放小水池预览图的文件夹
    PREVIEW_DIR = r"D:\Doc\博士毕业课题\Dataset_Previews"

    export_pool_previews(INPUT_DIR, PREVIEW_DIR)
    print("\n所有小水池预览图导出完毕！请在 ENVI 中查看。")
