import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
from tqdm import tqdm


# ==========================================
# 1. 预训练专属：无标签多光谱切片加载器
# ==========================================
class OilSpillH5Dataset(Dataset):
    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = h5_path

        datasets_info = []
        total_samples = 0
        with h5py.File(self.h5_path, 'r') as h5f:
            for group_name in h5f.keys():
                group = h5f[group_name]
                for class_name in group.keys():
                    dset_path = f"{group_name}/{class_name}"
                    num = h5f[dset_path].shape[0]
                    datasets_info.append((dset_path, num))
                    total_samples += num

        self.total_samples = total_samples
        print(f"成功扫库，共计检测到 {self.total_samples} 个多光谱 Patch。")

        print("🚀 [无监督] 正在将完整数据集载入 RAM 并执行全局归一化...")
        self.ram_data = np.empty((self.total_samples, 30, 16, 16), dtype=np.float32)

        curr_idx = 0
        with h5py.File(self.h5_path, 'r') as h5f:
            for dset_path, num in tqdm(datasets_info, desc="📥 载入内存"):
                raw_chunk = h5f[dset_path][:]
                raw_chunk = np.transpose(raw_chunk, (0, 3, 1, 2))
                self.ram_data[curr_idx: curr_idx + num] = raw_chunk
                curr_idx += num

        # 💡 核心抢救：处理负数，执行全局 Min-Max 归一化
        global_min = self.ram_data.min()
        global_max = self.ram_data.max()
        print(f"🌍 原始数据范围: Min={global_min:.4f}, Max={global_max:.4f}")

        if global_max - global_min > 1e-6:
            self.ram_data = (self.ram_data - global_min) / (global_max - global_min)

        print(f"✅ 归一化完成！当前范围: Min={self.ram_data.min():.4f}, Max={self.ram_data.max():.4f}")

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return torch.from_numpy(self.ram_data[idx])


# ==========================================
# 2. 下游任务专属：4分类有标签数据集加载器 (智能关键词匹配版)
# ==========================================
class OilSpill4ClassDataset(Dataset):
    def __init__(self, h5_path):
        super().__init__()
        self.h5_path = h5_path

        self.class_names = ["纯水 (Pure Water)", "汽油 (Gasoline)", "柴油 (Diesel)", "矿物油 (Mineral Oil)"]
        self.data_list = []
        self.label_list = []

        print("🚀 [4分类任务] 正在载入带标签的数据集并执行全局归一化...")
        with h5py.File(self.h5_path, 'r') as h5f:
            for group_name in h5f.keys():
                group = h5f[group_name]
                for class_name in group.keys():

                    # 💡 核心修改：将文件夹名转为小写，使用关键词模糊捕捉，绝对不会漏掉任何数据！
                    name_lower = class_name.lower()
                    matched_label = -1

                    if 'water' in name_lower or '纯水' in name_lower:
                        matched_label = 0
                    elif 'gasoline' in name_lower or '汽油' in name_lower:
                        matched_label = 1
                    elif 'diesel' in name_lower or '柴油' in name_lower:
                        matched_label = 2
                    elif 'mineral' in name_lower or '矿物' in name_lower:
                        matched_label = 3

                    # 如果成功匹配到了这 4 大类中的一种
                    if matched_label != -1:
                        dset_path = f"{group_name}/{class_name}"
                        raw_data = h5f[dset_path][:]
                        # 转换通道顺序 (N, 30, 16, 16)
                        raw_data = np.transpose(raw_data, (0, 3, 1, 2))
                        self.data_list.append(raw_data)
                        self.label_list.append(np.full(len(raw_data), matched_label, dtype=np.int64))

        if len(self.data_list) == 0:
            raise ValueError("❌ 没有匹配到任何数据，请检查 HDF5 数据库！")

        self.ram_data = np.concatenate(self.data_list, axis=0)
        self.ram_labels = np.concatenate(self.label_list, axis=0)

        # 💡 同步执行归一化，保持和预训练严格一致
        global_min = self.ram_data.min()
        global_max = self.ram_data.max()
        if global_max - global_min > 1e-6:
            self.ram_data = (self.ram_data - global_min) / (global_max - global_min)

        print(f"✅ 4 分类数据集载入完毕！本次成功读取 {len(self.ram_data)} 个切片。")
        for idx, name in enumerate(self.class_names):
            count = np.sum(self.ram_labels == idx)
            print(f"   -> [{name}]: {count} 个样本 ({count / len(self.ram_data) * 100:.1f}%)")

    def __len__(self):
        return len(self.ram_data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.ram_data[idx]), self.ram_labels[idx]