import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import matplotlib.pyplot as plt

# ─── CONFIGURATION ────────────────────────────────────────
DATASET_ROOT  = "/home/rooto1/Offroad_Segmentation_Training_Dataset"
NUM_CLASSES   = 10
IMAGE_SIZE    = 512
BATCH_SIZE    = 4 
EPOCHS        = 5      # Short run to generate visible "After" results 
SAMPLE_LIMIT  = 50     # Subset for immediate results
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIS_DIR       = "presentation_visuals"

os.makedirs(VIS_DIR, exist_ok=True)

# Class ID Mapping [cite: 35]
ID_MAP = {100:0, 200:1, 300:2, 500:3, 550:4, 600:5, 700:6, 800:7, 7100:8, 10000:9}
CLASS_NAMES = ["Trees","Bushes","Grass","DryBush","Clutter","Flowers","Logs","Rocks","Ground","Sky"]

# ─── DATA UTILS ───────────────────────────────────────────
def remap_mask(mask_array):
    out = np.zeros_like(mask_array, dtype=np.uint8)
    for raw_id, new_id in ID_MAP.items():
        out[mask_array == raw_id] = new_id
    return out

class VisualDataset(Dataset):
    def __init__(self, img_dir, msk_dir, transform=None, limit=50):
        self.img_dir, self.msk_dir = img_dir, msk_dir
        self.transform = transform
        all_files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
        random.seed(42)
        self.files = random.sample(all_files, limit)

    def __len__(self): return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        img_path = os.path.join(self.img_dir, fname)
        mask_path = os.path.join(self.msk_dir, fname)
        
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = remap_mask(np.array(Image.open(mask_path)))
        
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"].long()
        return img, mask, fname

# ─── AUGMENTATIONS ────────────────────────────────────────
train_transform = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE), # Fixed tuple error
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ─── VISUALIZATION GENERATOR ──────────────────────────────
def save_comparison(model, dataset, index, epoch):
    model.eval()
    img_tensor, mask_tensor, fname = dataset[index]
    
    with torch.no_grad():
        input_batch = img_tensor.unsqueeze(0).to(DEVICE)
        output = model(input_batch)
        pred = output.argmax(dim=1).squeeze().cpu().numpy()
    
    # Convert normalized tensor back to viewable image
    img_view = img_tensor.permute(1, 2, 0).numpy()
    img_view = (img_view * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]).clip(0, 1)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original Image")
    plt.imshow(img_view)
    
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth (Before)")
    plt.imshow(mask_tensor.numpy(), cmap='nipy_spectral')
    
    plt.subplot(1, 3, 3)
    plt.title(f"Model Prediction (After Epoch {epoch})")
    plt.imshow(pred, cmap='nipy_spectral')
    
    plt.savefig(f"{VIS_DIR}/comparison_epoch_{epoch}_{fname}")
    plt.close()

# ─── TRAINING ─────────────────────────────────────────────
def main():
    train_img = os.path.join(DATASET_ROOT, "train", "Color_Images")
    train_msk = os.path.join(DATASET_ROOT, "train", "Segmentation")

    train_ds = VisualDataset(train_img, train_msk, train_transform, limit=SAMPLE_LIMIT)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # Core Model: DeepLabV3+ with ResNet-50 [cite: 10, 15]
    model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet", 
                              in_channels=3, classes=NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4) # [cite: 25]
    criterion = smp.losses.DiceLoss(mode="multiclass") # [cite: 39]

    print(f"Generating 'Before/After' visuals for 50 images...")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for imgs, masks, _ in tqdm(train_loader, desc=f"Epoch {epoch}"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward(); optimizer.step()

        # Save visual result after each epoch to show improvement
        save_comparison(model, train_ds, index=0, epoch=epoch)
        print(f"Snapshot saved to {VIS_DIR}")

    print(f"Visuals generated. Check the '{VIS_DIR}' folder for your PPT slides.")

if __name__ == "__main__":
    main()
