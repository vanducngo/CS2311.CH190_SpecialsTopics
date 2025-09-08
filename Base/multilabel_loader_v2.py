import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
import os
from PIL import Image

from .metrics import AUCProcessor
from .corruptions import apply_corruption

class ContinualCorruptionDataset(Dataset):
    def __init__(self, cfg, disable_normalize=False):
        # 1. Đọc cấu hình
        base_domain_cfg = cfg.DATASET.BASE_DOMAIN
        self.corruptions_to_apply = cfg.DATASET.TEST_CORRUPTIONS if cfg.DATASET.TEST_CORRUPTIONS else ['none']
        self.severity = cfg.DATASET.SEVERITY
        self.labels_list = cfg.DATASET.LABELS_LIST
        self.disable_normalize = disable_normalize

        # 2. Tải dữ liệu "sạch" gốc
        self.root_dir = os.path.join(base_domain_cfg.PATH, base_domain_cfg.IMAGE_DIR)
        csv_path = os.path.join(base_domain_cfg.PATH, base_domain_cfg.CSV)
        print(f"Loading base data from: {csv_path}")
        self.df = pd.read_csv(csv_path)
        self.image_col = 'image_id'
        print(f"Loaded {len(self.df)} base samples.")

        # 3. Định nghĩa các transform cơ bản
        self.to_tensor_transform = transforms.ToTensor()
        self.normalize_transform = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.resize_transform = transforms.Resize((224, 224))

        # 4. Tạo danh sách mẫu lớn, được sắp xếp theo từng loại nhiễu
        self.all_samples = []
        print("Building continual corruption stream...")
        # Vòng lặp ngoài là CORRUPTION
        for corruption_name in self.corruptions_to_apply:
            # Vòng lặp trong là TẤT CẢ các ảnh
            for i in range(len(self.df)):
                self.all_samples.append((i, corruption_name))
                
        print(f"Continual Corruption Dataset created with {len(self.all_samples)} total samples.")
        print(f"Corruption sequence: {self.corruptions_to_apply}")

    def __len__(self):
        return len(self.all_samples)

    def __getitem__(self, idx):
        # Lấy chỉ số ảnh gốc và tên nhiễu từ danh sách đã tạo
        sample_idx, corruption_name = self.all_samples[idx]
        
        row = self.df.iloc[sample_idx]
        img_name = str(row[self.image_col])
        
        if ("CheXpert-v1.0-small" in img_name):
            img_path = f"/home/ngoto/Working/Data/{img_name}"
        elif not os.path.isabs(img_name):
             img_path = os.path.join(self.root_dir, img_name)
        else:
             img_path = img_name

        try:
            image_pil = self.resize_transform(Image.open(img_path).convert('RGB'))
        except FileNotFoundError:
            print(f"Warning: File not found at {img_path}. Returning None.")
            return None

        image_tensor = self.to_tensor_transform(image_pil)
        corrupted_tensor = apply_corruption(image_tensor, corruption_name, self.severity)
        
        if not self.disable_normalize:
            final_image = self.normalize_transform(corrupted_tensor)
        else:
            final_image = corrupted_tensor 

        labels = torch.tensor(row[self.labels_list].values.astype('float'), dtype=torch.float32)
        
        return {'image': final_image, 'label': labels, 'domain': corruption_name}

def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return {'image': torch.empty(0), 'label': torch.empty(0), 'domain': []}
    return torch.utils.data.dataloader.default_collate(batch)

def build_loader_multilabel(cfg):
    continual_corruption_dataset = ContinualCorruptionDataset(cfg)
    loader = DataLoader(
        continual_corruption_dataset,
        batch_size=cfg.TEST.BATCH_SIZE,
        shuffle=False,  # Cực kỳ quan trọng: Phải là False để giữ đúng thứ tự nhiễu
        num_workers=cfg.LOADER.NUM_WORKS,
        collate_fn=collate_fn_skip_none,
        pin_memory=True
    )
    
    result_processor = AUCProcessor(num_classes=len(cfg.DATASET.LABELS_LIST))
    
    print("Continual corruption data loader and AUC processor built successfully.")
    return loader, result_processor