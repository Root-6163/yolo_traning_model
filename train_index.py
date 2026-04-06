import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ─── CONFIGURATION ────────────────────────────────────────
DATASET_ROOT  = "/home/rooto1/Offroad_Segmentation_Training_Dataset"
NUM_CLASSES   = 10
IMAGE_SIZE    = 512
BATCH_SIZE    = 4
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BEFORE_DIR = "Before"
AFTER_DIR  = "After"
os.makedirs(BEFORE_DIR, exist_ok=True)
os.makedirs(AFTER_DIR, exist_ok=True)

#  VISUAL COLOR PALETTE (Solves the "Black Image" problem)
COLOR_MAP = [
    128, 0, 0, 0, 128, 0, 128, 128, 0, 0, 0, 128,
    128, 0, 128, 0, 128, 128, 192, 192, 192,
    64, 64, 64, 255, 255, 255, 0, 191, 255
] + [0] * (768 - 30)

def save_as_colored_png(array, path):
    """Saves a 0-9 index array as a visible colored PNG[cite: 51]."""
    img = Image.fromarray(array.astype(np.uint8), mode='P')
    img.putpalette(COLOR_MAP)
    img.save(path)

# ─── DATA & REMAPPING ─────────────────────────────────────
ID_MAP = {100:0, 200:1, 300:2, 500:3, 550:4, 600:5, 700:6, 800:7, 7100:8, 10000:9}

def remap_mask(mask_array):
    """Converts raw simulator IDs to 0-9 indices[cite: 35]."""
    out = np.zeros_like(mask_array, dtype=np.uint8)
    for raw_id, new_id in ID_MAP.items():
        out[mask_array == raw_id] = new_id
    return out

class FastDataset(Dataset):
    def __init__(self, img_dir, msk_dir, transform=None):
        self.img_dir, self.msk_dir = img_dir, msk_dir
        self.transform = transform
        self.files = sorted([f for f in os.listdir(img_dir) if f.endswith(".png")])
    def __len__(self): return len(self.files)
    def __getitem__(self, idx):
        fname = self.files[idx]
        img = np.array(Image.open(os.path.join(self.img_dir, fname)).convert("RGB"))
        mask = remap_mask(np.array(Image.open(os.path.join(self.msk_dir, fname))))
        if self.transform:
            aug = self.transform(image=img, mask=mask)
            img, mask = aug["image"], aug["mask"].long()
        return img, mask, fname

transform = A.Compose([
    A.Resize(height=IMAGE_SIZE, width=IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ─── TRAINING & VISUALIZATION ─────────────────────────────
def main():
    train_img = os.path.join(DATASET_ROOT, "train", "Color_Images")
    train_msk = os.path.join(DATASET_ROOT, "train", "Segmentation")
    dataset = FastDataset(train_img, train_msk, transform)

    # --- USER INPUT FOR INDEXING ---
    try:
        start_idx = int(input("Enter starting image index (e.g., 2): "))
        end_idx   = int(input("Enter ending image index (e.g., 30): "))
        
        # Ensure indices are within dataset range
        start_idx = max(0, start_idx)
        end_idx = min(len(dataset), end_idx)
        
        if start_idx >= end_idx:
            print("Error: Start index must be smaller than end index.")
            return
    except ValueError:
        print("Invalid input. Please enter integers.")
        return

    indices = list(range(start_idx, end_idx))
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True)

    # Core Model: DeepLabV3+ with ResNet-50 [cite: 9, 10]
    model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights="imagenet",
                              in_channels=3, classes=NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = smp.losses.DiceLoss(mode="multiclass")

    print(f" Training on {len(subset)} images (Index {start_idx} to {end_idx-1})...")
    for epoch in range(1, 11):
        model.train()
        for imgs, masks, _ in tqdm(loader, desc=f"Epoch {epoch}"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(imgs), masks)
            loss.backward(); optimizer.step()

    model.eval()
    print(f" Saving colored results to /{BEFORE_DIR} and /{AFTER_DIR}...")
    with torch.no_grad():
        for i in range(len(subset)):
            img_t, msk_t, fname = subset[i]
            # Save "Before" (Ground Truth)
            save_as_colored_png(msk_t.numpy(), os.path.join(BEFORE_DIR, f"gt_{fname}"))
            # Save "After" (Prediction)
            pred = model(img_t.unsqueeze(0).to(DEVICE)).argmax(dim=1).squeeze().cpu().numpy()
            save_as_colored_png(pred, os.path.join(AFTER_DIR, f"pred_{fname}"))

    print(f" Success! {len(subset)} images processed and saved.")

if __name__ == "__main__":
    main()
