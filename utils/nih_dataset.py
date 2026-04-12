import os
import pandas as pd
import numpy as np


def compute_pos_weights_from_csv(csv_path, label_names, fallback_pos_weight=5.0):
    """
    Read train.csv and compute per-class positive-to-negative weight ratios for BCEWithLogitsLoss.

    For each class: pos_weight = num_negatives / num_positives (PyTorch convention).

    Returns:
        dict[str, float]: class name -> pos_weight
    """
    df = pd.read_csv(csv_path)
    weights = {}
    for name in label_names:
        if name not in df.columns:
            weights[name] = float(fallback_pos_weight)
            continue
        col = df[name]
        n_pos = int((col == 1).sum())
        n_neg = int((col == 0).sum())
        if n_pos == 0:
            weights[name] = float(fallback_pos_weight)
        else:
            weights[name] = float(n_neg) / float(n_pos)
    return weights
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def prepare_nih_csv(train_csv_path, valid_csv_path, data_dir):
    train_df = pd.read_csv(train_csv_path)
    val_df = pd.read_csv(valid_csv_path)

    # Target 5 labels mapping from REFLACX to NIH
    target_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Effusion']

    # The paths in the CSV start with "images/" (e.g., "images/00000001_000.png")
    # data_dir is expected to be the directory containing the "images" folder
    train_df['image_path'] = train_df['Path'].apply(lambda x: os.path.join(data_dir, x))
    val_df['image_path'] = val_df['Path'].apply(lambda x: os.path.join(data_dir, x))

    return train_df, val_df, target_labels

class NIHChestDataset(Dataset):
    def __init__(self, df, target_labels, transform=None):
        self.df = df
        self.target_labels = target_labels
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Error loading image {image_path}: {e}. Using blank image.")
            image = Image.new('RGB', (256, 256))

        if self.transform:
            image = self.transform(image)

        labels = row[self.target_labels].values.astype(np.float32)
        return image, torch.tensor(labels)

def get_nih_dataloaders(train_csv_path, valid_csv_path, data_dir, batch_size=16, num_workers=0):
    train_df, val_df, target_labels = prepare_nih_csv(train_csv_path, valid_csv_path, data_dir)

    train_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomCrop(112),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = NIHChestDataset(train_df, target_labels, transform=train_transform)
    val_dataset = NIHChestDataset(val_df, target_labels, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True,
        persistent_workers=(num_workers > 0)
    )

    return train_loader, val_loader
