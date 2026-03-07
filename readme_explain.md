# Multimodal Medical Diagnosis System: NIH ChestX-ray14 Adaptation

This repository is an adaptation of the original Multimodal Medical Diagnosis System. While the original system was designed to leverage MIMIC-IV and REFLACX clinical triage data alongside X-rays, this specific adaptation strips away the clinical data dependencies to demonstrate pure image-based classification using the **NIH ChestX-ray14** dataset.

## 🎯 Clinical Objective
The goal of this adaptation is to train a high-speed, image-only Deep Learning vision model (DenseNet121) capable of detecting 5 major cardiopulmonary diseases from frontal chest X-rays (AP/PA):
- **Atelectasis**
- **Cardiomegaly**
- **Consolidation**
- **Edema**
- **Effusion**

## 📂 Dataset Preprocessing
The model operates on a subset of the massive NIH ChestX-ray14 dataset, utilizing roughly **98,637 training samples** and **6,350 validation samples**.
* **Loader Location:** `utils/nih_dataset.py`
* The dataset `.tar.gz` files are extracted to `data/nih_chest_xray/images/`.
* The project leverages pre-parsed CSV files (`cxr_code/data/train.csv` and `valid.csv`) to map image paths directly to multi-label binarized vectors for the 5 target findings.

To drastically increase training speeds, the images are automatically down-sampled to `128x128` pixels natively inside the dataloader while preserving critical chest cavity ratios via Random & Center Cropping.

## ⚙️ GPU Acceleration & Environment Setup
Due to a lack of native Windows CUDA pre-compilations for Python 3.14+, this project relies on an isolated Python 3.12 virtual environment (`venv_gpu`) tailored specifically for raw GPU throughput.
* **PyTorch Version:** 2.6.x (with `--index-url https://download.pytorch.org/whl/cu121`)
* **Hardware:** Native NVIDIA CUDA bindings

## 🚀 Training the Model
* **Script:** `scripts/train_nih.py`
* **Network:** `ImageOnlyModel` wrapper around the pre-trained `torchvision` models.
* **Loss Function:** `nn.BCEWithLogitsLoss`. Target classes are highly imbalanced (e.g., most X-rays have 'No Finding'). `pos_weight` tensors are applied natively to penalize false negatives for minority classes aggressively.

### Extreme Performance Optimizations
Training nearly 100,000 images per epoch can bottleneck a local machine. Extreme optimizations have been baked into the training script:
1. **Automatic Mixed Precision (AMP):** Utilizes `torch.amp.GradScaler` and `@autocast('cuda')` to force the GPU's Tensor Cores to train the network using 16-bit floating point precision matrices where possible, effectively doubling native compute speeds.
2. **Aggressive Batch Sizing:** Increased from 16 to 128 elements per pass to fully saturate modern GPU video memory bandwidth.
3. **Persistent multiprocessing:** Overcame Windows DataLoader deadlocks by launching `num_workers=4` processes and `pin_memory=True`. The CPU decodes the PNG files on separate threads and physically locks the batches into RAM ahead of the GPU, virtually eliminating I/O stalls.

**Run the training script:**
```bash
.\venv_gpu\Scripts\python.exe scripts/train_nih.py
```
*The model evaluates itself via Mean Reciever-Operating Characteristic Area Under Curve (ROC-AUC) measurements at the end of each epoch and writes the best configuration to `results/best_model.pth`.*

## 🔍 Running Inference
Once trained, the `scripts/predict_nih.py` script allows you to diagnose any native `.png` or `.jpg` image instantly.

The script operates in a purely `torch.no_grad()` autograd-isolated context and mimics the original repository's output format. It will parse the image, evaluate the 5 diseases, log the output to the console, and export a discrete `.txt` report alongside a visual `.png` Bar Plot of the confidence intervals.

**Run the inference script:**
```bash
.\venv_gpu\Scripts\python.exe scripts/predict_nih.py test.png
```
*Artifacts are automatically exported to `results/[image_name]_inference/`*
