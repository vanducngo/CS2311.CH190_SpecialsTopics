import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import torchvision.transforms.functional as TF

from Base.constants import COMMON_FINAL_LABEL_SET, TRAINING_LABEL_SET

class QuantizationAugmentation:
    def __call__(self, tensor):
        pil_img = TF.to_pil_image(tensor)
        return TF.to_tensor(pil_img)

class CheXpertFullLabelDataset(Dataset):
    def __init__(self, cfg, mode='train', transform=None):
        self.cfg = cfg
        self.transform = transform
        
        csv_file = cfg.DATA.CHEXPERT_TRAIN_CSV if mode in ['train', 'valid'] else cfg.DATA.CHEXPERT_TEST_CSV
        csv_path = os.path.join(cfg.DATA.CHEXPERT_PATH, csv_file)
        
        print(f'csv_file for {mode}: {csv_file}')

        raw_df = pd.read_csv(csv_path)
        self.df = raw_df
        self.root_dir = cfg.DATA.CHEXPERT_PATH_ROOT_PATH
        print(f"Loaded CheXpert {mode} dataset with {len(self.df)} samples for {len(TRAINING_LABEL_SET)} classes.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.df.iloc[idx]['image_id'])
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return None

        labels = self.df.iloc[idx][TRAINING_LABEL_SET].values.astype('float')
        labels = torch.tensor(labels, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
        
        return image, labels
    
class MultiSourceDataset(Dataset):
    def __init__(self, cfg, dataset_name, mode='train', transform=None):
        self.cfg = cfg
        self.dataset_name = dataset_name
        self.mode = mode
        self.transform = transform
        
        if self.dataset_name == 'chexpert':
            csv_path = os.path.join(cfg.DATA.CHEXPERT_PATH, 
                                    cfg.DATA.CHEXPERT_TRAIN_CSV if mode == 'train' else cfg.DATA.CHEXPERT_TEST_CSV)
            self.df = pd.read_csv(csv_path)
            self.root_dir = cfg.DATA.CHEXPERT_PATH_ROOT_PATH
            self.image_col = 'image_id'
        elif self.dataset_name == 'nih14':
            csv_path = os.path.join(cfg.DATA.NIH14_PATH, cfg.DATA.NIH14_CSV)
            self.df = pd.read_csv(csv_path)
            self.root_dir = os.path.join(cfg.DATA.NIH14_PATH, cfg.DATA.NIH14_IMAGE_DIR) 
            self.image_col = 'image_id'
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")
            
        print(f"Loaded and mapped {self.dataset_name} ({mode}) with {len(self.df)} samples.")
        print(f"Common diseases being used: {COMMON_FINAL_LABEL_SET}")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx][self.image_col]

        path_prefix = '' if (img_name.endswith('.png') or img_name.endswith('.jpg')) else '.png'
        img_path = os.path.join(self.root_dir, img_name + path_prefix)
            
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: File not found at {img_path}. Skipping.")
            return torch.empty(0), torch.empty(0)
        except Exception as e:
            print(f"Error reading {img_path}: {e}. Skipping.")
            return torch.empty(0), torch.empty(0)

        # Get labels from common disease columns
        labels = self.df.iloc[idx][COMMON_FINAL_LABEL_SET].values.astype('float')
        labels = torch.tensor(labels, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels

def collate_fn(batch):
    batch = list(filter(lambda x: x[0].numel() > 0, batch))
    if not batch:
        return torch.empty(0), torch.empty(0)
    return torch.utils.data.dataloader.default_collate(batch)

def get_chexpert_full_label_loaders(cfg):   
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transformTrain = transforms.Compose([
        transforms.Resize((224, 224)),
        
        transforms.RandomRotation(15), 
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        QuantizationAugmentation(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = CheXpertFullLabelDataset(cfg, mode='train', transform=transformTrain)
    train_subset = torch.utils.data.Subset(train_dataset, range(len(train_dataset)))
    train_loader = DataLoader(train_subset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    
    chexpert_test_dataset = CheXpertFullLabelDataset(cfg, mode='test', transform=transform)
    chexpert_test_loader = DataLoader(chexpert_test_dataset, batch_size=cfg.TRAINING.BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=4)

    return train_loader, chexpert_test_loader