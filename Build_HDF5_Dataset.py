import numpy as np
import spectral.io.envi as envi
from scipy.signal import savgol_filter
import h5py
from pathlib import Path
from tqdm import tqdm


# ==========================================
# 1. 物理清洗模块
# ==========================================
def remove_sun_glint(reflectance_cube, nir_band_idx=29, target_nir=0.01):
    glint_corrected = np.copy(reflectance_cube)
    nir_band = reflectance_cube[:, :, nir_band_idx]
    delta_glint = nir_band - target_nir
    delta_glint[delta_glint < 0] = 0
    for i in range(reflectance_cube.shape[2]):
        glint_corrected[:, :, i] -= delta_glint
    return np.clip(glint_corrected, 0, 1)


def generate_water_mask(reflectance_cube, green_idx=7, nir_idx=29, threshold=0.0):
    green = reflectance_cube[:, :, green_idx]
    nir = reflectance_cube[:, :, nir_idx]
    ndwi = (green - nir) / (green + nir + 1e-8)
    mask = np.zeros_like(green, dtype=np.uint8)
    mask[ndwi > threshold] = 1
    return mask


# ==========================================
# 2. HDF5 切片保存模块
# ==========================================
def save_patches_to_hdf5(cleaned_roi, mask, h5_file_obj, dataset_name, patch_size=16, stride=8):
    h, w, c = cleaned_roi.shape
    patch_list = []

    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            mask_patch = mask[i:i + patch_size, j:j + patch_size]

            # 纯净度检查：超过 90% 面积是有效水面才保留
            valid_ratio = np.sum(mask_patch) / (patch_size * patch_size)
            if valid_ratio < 0.90:
                continue

            patch = cleaned_roi[i:i + patch_size, j:j + patch_size, :]

            # 滤除全黑死像素
            if np.mean(patch) < 0.01:
                continue

            patch_list.append(patch)

    if not patch_list: return 0

    patch_array = np.stack(patch_list).astype(np.float32)

    if dataset_name in h5_file_obj:
        dset = h5_file_obj[dataset_name]
        dset.resize(dset.shape[0] + patch_array.shape[0], axis=0)
        dset[-patch_array.shape[0]:] = patch_array
    else:
        h5_file_obj.create_dataset(dataset_name, data=patch_array, maxshape=(None, patch_size, patch_size, c),
                                   chunks=True)
    return len(patch_list)


# ==========================================
# 3. 从预裁剪文件构建 HDF5 主流水线
# ==========================================
def build_hdf5_from_cropped(input_dir, output_h5_path):
    root_path = Path(input_dir)
    # 自动获取目录下所有的类别文件夹 (例如 00_PureWater, 01_Gasoline92_10um 等)
    class_folders = [f for f in root_path.iterdir() if f.is_dir()]

    print(f"检测到 {len(class_folders)} 个油种类别文件夹，准备开始打包切片...")

    with h5py.File(output_h5_path, 'a') as h5f:
        for folder in class_folders:
            class_name = folder.name
            dataset_name = f"All_Altitudes/{class_name}"

            # 在当前类别文件夹下，搜寻所有完整 30 波段的数据文件
            fb_files = list(folder.rglob("*_FullBands.dat"))
            if not fb_files:
                continue

            for fb_path in tqdm(fb_files, desc=f"打包类 [{class_name}]"):
                # 1. 读取完整的 30 波段数据
                fb_hdr = fb_path.with_suffix('.hdr')
                img_fb = envi.open(fb_hdr, fb_path)
                roi = np.array(img_fb.load(), dtype=np.float32)

                # 2. 构造并读取对应的精准掩码文件
                mask_path = fb_path.parent / fb_path.name.replace("_FullBands.dat", "_Mask.dat")
                if not mask_path.exists():
                    print(f"警告：找不到对应的掩码文件 {mask_path.name}，跳过此图。")
                    continue

                mask_hdr = mask_path.with_suffix('.hdr')
                img_mask = envi.open(mask_hdr, mask_path)
                # 掩码读取出来可能是 (H, W, 1)，通过 squeeze 降维成 (H, W)
                exact_pool_mask = np.array(img_mask.load(), dtype=np.float32).squeeze()

                # 3. 光谱清洗与平滑
                roi_glint = remove_sun_glint(roi, nir_band_idx=29)
                roi_smoothed = savgol_filter(roi_glint, window_length=5, polyorder=2, axis=2)

                # 4. 生成水体 NDWI 掩码 (过滤池壁杂质)
                water_mask = generate_water_mask(roi_smoothed, green_idx=7, nir_idx=29)

                # 5. 双重保险掩码：既要是水，又必须在橙色网格内部
                final_mask = (exact_pool_mask == 1) & (water_mask == 1)

                # 6. 滑动切片并压入 HDF5
                save_patches_to_hdf5(roi_smoothed, final_mask, h5f, dataset_name)


if __name__ == "__main__":
    # 输入：之前成功导出的小水池预览文件夹（包含 10 个类别子文件夹）
    INPUT_DIR = r"D:\Doc\博士毕业课题\Dataset_Previews"

    # 输出：最终用于训练的 HDF5 数据库文件
    OUTPUT_H5 = r"E:\Dsw\Dataset\OilSpill_Dataset.h5"

    Path(OUTPUT_H5).parent.mkdir(parents=True, exist_ok=True)

    # 注意：如果多次运行测试，建议先手动删除现有的 OilSpill_Dataset.h5 以防数据重复追加
    build_hdf5_from_cropped(INPUT_DIR, OUTPUT_H5)
    print("\n恭喜！预处理的最后一步完成，HDF5 深度学习数据库已成功生成！")