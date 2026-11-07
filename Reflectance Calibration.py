import numpy as np
import spectral.io.envi as envi
from scipy.interpolate import interp1d
from pathlib import Path
from tqdm import tqdm


# ==========================================
# 1. 初始化白板的绝对物理系数
# ==========================================
def get_sensor_cloth_reflectance(sensor_wavelengths):
    """
    将厂家提供的白板系数插值到你相机的 30 个波段上
    """
    # 【请替换为你真实的厂家数据】
    # 假设厂家提供的数据是：波长 (nm) 和 对应的反射率 (0~1)
    cloth_waves = np.array([400,410,420,430,440,450,460,470,480,490,
                            500,510,520,530,540,550,560,570,580,590,
                            600,610,620,630,640,650,660,670,680,690,
                            700,710,720,730,740,750,760,770,780,790,
                            800,810,820,830,840,850,860,870,880,890,
                            900,910,920,930,940,950,960,970,980,990,1000])
    cloth_refs = np.array([39.5338707322222,62.5512503111111,78.3942226233333,80.9423848566666,81.9483774766667,82.2457978144444,82.6180916,82.9327149933333,82.6839594444444,83.0497103311111,82.7991897377778,82.8877656333333,82.8281480166667,82.7591744477778,82.7487655966667,82.7579618955555,82.7131941811111,82.6853527488889,82.6392082766667,82.7060537066667,82.5381543555556,82.6449592677778,82.8500883955556,82.81195142,82.8386371233333,82.7828781855555,82.7872276133333,82.5405840922222,82.5270960588889,82.6286453044444,82.6234266544444,82.6395275655555,82.5859168466666,82.3690451566667,82.42100178,82.31346761,82.1894998988889,82.0780232744444,81.9673366933333,81.9185085288889,81.7134108522222,81.6258824577778,81.3154672666666,81.2337455866667,81.0759118955555,80.9548775888889,80.9815621222222,80.6787855977778,80.6038002411111,80.5990915322222,80.26637361,80.2444546244444,80.0707146444444,80.0918236688889,79.9205679222222,79.9426423722222,79.9060993233333,79.6924993066667,79.7566274755556,79.7916006044444,79.5041416844444]) / 100.0

    # 使用三次样条插值 (Cubic Spline) 进行平滑拟合
    interp_func = interp1d(cloth_waves, cloth_refs, kind='cubic', fill_value="extrapolate")

    # 计算你相机 30 个波段对应的理论反射率
    sensor_cloth_refs = interp_func(sensor_wavelengths)
    return sensor_cloth_refs  # 形状为 (30,)


# ==========================================
# 2. 核心：自动化白板提取算法
# ==========================================
def auto_extract_white_panel_dn(data_cube):
    """
    从影像中自动定位白板，并返回 30 个波段的平均 DN 值
    """
    # 1. 计算全波段平均亮度
    brightness = np.mean(data_cube, axis=2)

    # 2. 找到极亮区域 (假设白板占画面的极小部分，取 99.5% 分位数以上的像素)
    threshold = np.percentile(brightness, 99.5)

    # 3. 生成初步掩码：比阈值亮，且没有因为反光过曝(DN<65000, 假设是16bit)
    # 注意：根据你的相机的实际过曝值(DN max)调整 65000 这个数字
    bright_mask = (brightness > threshold) & (np.max(data_cube, axis=2) < 65000)

    if np.sum(bright_mask) < 10:
        raise ValueError("未能在这张图像中自动找到足够大的白板区域，请检查阈值或图像！")

    # 4. 提取这些像素的光谱曲线，并计算平均值
    # 得到该图像在当时太阳光照下的 白板参考 DN 值
    white_panel_dn_mean = np.mean(data_cube[bright_mask], axis=0)

    return white_panel_dn_mean  # 形状为 (30,)


# ==========================================
# 3. 自动化批处理流水线
# ==========================================
def auto_radiometric_calibration_pipeline(data_dir, output_dir):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    spe_files = list(Path(data_dir).rglob("*.spe"))
    print(f"总共在子文件夹中找到了 {len(spe_files)} 个 .spe 影像组。")

    # 从你的 hdr 中提取的波长 (之前你提供的)
    sensor_waves = np.array([
        409.08, 431, 450.58, 470.02, 489.34, 510.65, 529.72, 550.77, 569.62, 590.45,
        609.11, 629.76, 650.32, 670.8, 689.18, 709.55, 729.87, 750.15, 770.41, 790.65,
        810.88, 829.09, 849.34, 869.6, 889.9, 910.24, 930.62, 949.02, 969.53, 990.11
    ])

    # 获取波段对齐后的白板理论反射率系数 (30,)
    cloth_refs_30bands = get_sensor_cloth_reflectance(sensor_waves)

    for spe_path in tqdm(spe_files, desc="执行自动辐射定标"):
        hdr_path = spe_path.with_suffix('.hdr')
        img = envi.open(hdr_path, spe_path)

        data = img.load()
        data = np.array(data, dtype=np.float32)

        # 1. 自动寻找当天的白板 DN 值
        white_dn = auto_extract_white_panel_dn(data)

        # 2. 全局辐射校正
        data_calibrated = (data / (white_dn + 1e-8)) * cloth_refs_30bands
        data_calibrated = np.clip(data_calibrated, 0, 1)

        # ==============================================================
        # 3. 新增保存模块：将定标后的物理反射率保存为新的 ENVI 文件
        # ==============================================================
        relative_path = spe_path.relative_to(data_dir)
        save_spe_path = Path(output_dir) / relative_path
        save_spe_path.parent.mkdir(parents=True, exist_ok=True)

        # 构造保存文件的元数据
        metadata = img.metadata.copy()
        metadata['data type'] = 4  # ENVI float32

        # 强制把 hdr 头文件里的交错格式改为 bsq
        metadata['interleave'] = 'bsq'

        save_hdr_path = save_spe_path.with_suffix('.hdr')

        envi.save_image(
            str(save_hdr_path),
            data_calibrated,
            metadata=metadata,
            dtype=np.float32,
            interleave='bsq',
            force=True,
            ext='.spe'
        )

        # (可选) 显式清理内存，对于处理数百个大文件是个好习惯
        del data, data_calibrated

if __name__ == "__main__":
     # 1. 设置输入路径：把你装有 .spe 和 .hdr 文件的文件夹路径填在这里
     # 注意：Windows 路径前面最好加一个字母 r，或者把斜杠改成双斜杠 \\，防止路径解析报错
     INPUT_DIR = r"E:\FigSpec Files"

     # 2. 设置输出路径：处理完的数据想要保存在哪个文件夹
     OUTPUT_DIR = r"D:\Doc\博士毕业课题\辐射校正"

     # 3. 执行流水线
     print(f"正在读取文件夹: {INPUT_DIR}")
     auto_radiometric_calibration_pipeline(INPUT_DIR, OUTPUT_DIR)
     print("辐射定标流程运行完毕！")