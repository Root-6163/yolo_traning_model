
# 🏜️ YOLO Training Model: Offroad Segmentation
**Project**: Duality AI Offroad Segmentation Challenge  
**Team**: Tech Sorcerer (Aditya Waje, Girish Unde, Gopal Rajane)

---

## 📋 Project Overview
This hackathon task involves training a semantic segmentation model on raw synthetic desert images from the Falcon simulator. The goal is to categorize pixels into 10 distinct classes to enable autonomous offroad navigation.

### **The Core Challenge: Data Transformation**
Raw simulator masks use high-value IDs (100–10,000) that cause training crashes. Our model implements a **0–9 remapping layer** to stabilize the pipeline:
* **Trees**: 100 → 0
* **Bushes**: 200/500 → 1/3
* **Rocks**: 800 → 7
* **Sky**: 10,000 → 9

---

## 💻 Setup & Execution (Linux)

### **1. Setup account and download offline data**
Ensure your Linux environment (Kali/Ubuntu) has the necessary runtime:

#### Set a account
https://falcon.duality.ai/auth/sign-up?utm_source=hackathon&utm_medium=instructions&utm_campaign=HacktheNight

```bash
#### download training data
wget https://falcon.duality.ai/secure/documentation/hackathon-segmentation-desert?utm_source=hackathon&utm_medium=instructions&utm_campaign=HacktheNight
```

### **1. Install Python**
Ensure your Linux environment (Kali/Ubuntu) has the necessary runtime:
```bash
sudo apt update
sudo apt install python3 python3-pip
pip install notebook  # Install Jupyter Notebook for .ipynb files
```

### **2. Install Dependencies**
Install the required deep learning and image processing libraries:
```bash
pip install -r requirements.txt
```
> **Note**: Requirements include `torch`, `segmentation-models-pytorch`, `albumentations`, `tqdm`, and `pillow`.

### **3. Run the Model**
Execute the training script to begin the segmentation process. This script is optimized to train on 100-image subsets for fast iteration:
```bash
python3 train.py
```

---

## 🛠️ Architecture & Strategy
* **Model**: DeepLabV3+ with a ResNet-50 backbone.
* **Pre-training**: Initialized with ImageNet weights, boosting mIoU from $\approx 0.45$ to $\approx 0.68$.
* **Loss Function**: Hybrid **Dice + Focal Loss** to handle class imbalance (e.g., rare "Flowers" vs. dominant "Sky").
* **Performance**: Optimized for real-time inference at **<50ms per image**.

---

##  Output & Submission
The script generates two critical folders for the final presentation:
* **`/Before`**: Target Ground Truth masks with a custom color palette for visibility.
* **`/After`**: Model-generated predictions showing sharp edge detection on Trees and Rocks.

---
