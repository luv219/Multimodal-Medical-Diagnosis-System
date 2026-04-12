# Running the NIH Dataset GPU Model

To run this model on a new machine, you need a highly specific Python environment. The training process has been heavily optimized for modern NVIDIA GPUs utilizing Automatic Mixed Precision, which requires native PyTorch CUDA bindings. 

Because official Python 3.13+ distributions currently have poor PyTorch Native CUDA wheel support on Windows, **you must use Python 3.12** for the environment to run perfectly.

## 1. Prerequisites (Windows)
1. Ensure your system has an **NVIDIA GPU** installed.
2. Ensure you have the latest **NVIDIA Studio or Game Ready Drivers** installed on your OS. 
3. Download and install **Python `3.12.x`** from python.org.
   * *During installation, ensure you check the box that says "Add Python 3.12 to PATH".*

## 2. Environment Setup
Open a Terminal (or PowerShell) at the root of the project directory.

**A. Create a Python 3.12 Virtual Environment**
Create an isolated environment (we'll call it `venv_gpu`) to prevent version conflicts with your main machine:
```bash
# If Python 3.12 is your default:
python -m venv venv_gpu

# OR, if you have multiple Pythons installed, force 3.12 via launcher:
py -3.12 -m venv venv_gpu
```

**B. Install Core Dependencies**
Before PyTorch, install the standard scientific processing dependencies using the new environment's pip:
```bash
.\venv_gpu\Scripts\python.exe -m pip install numpy pandas matplotlib scikit-learn pillow tqdm
```

**C. Install PyTorch + CUDA**
To train or inference the model natively on your graphics card at maximum speed, you must use the PyTorch index URL hosting the pre-compiled `cu121` (CUDA 12.1) `.whl` files for Python 3.12:
```bash
.\venv_gpu\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 3. Preparing the Dataset (Training Only)
If you just want to run inference, you can skip this step and use the `best_model.pth` file. If you wish to retrain the brain from scratch:
1. Download the massive **NIH ChestX-ray14** dataset from Kaggle or the NIH website.
2. Extract all the physical `.png` files from the `images_*.tar.gz` archives into the `data/nih_chest_xray/images/` folder inside this project. *(Ensure the folder is specifically named `/images/` as the CSV files map directly to that name)*.
3. The binarized metadata mapping the labels is already bundled inside the repository at `cxr_code/data/train.csv` and `valid.csv`.

## 4. Execution

### To Train the Model:
The system will automatically grab the GPU, initialize Mixed Precision Tensor Cores, lock batches into RAM via `persistent_workers`, and process the 100K images over 5 epochs. At the end of every epoch, it exports its memory to `results/best_model.pth`.
```bash
.\venv_gpu\Scripts\python.exe scripts/train_nih.py
```

### To Run Inference (Diagnosis):
To diagnose a single image on any machine, use the predict script. It will load `best_model.pth`, bypass the gradients, and output a raw diagnostic report and a confidence bar chart to the `results/` folder mirroring the image's name.
```bash
.\venv_gpu\Scripts\python.exe scripts/predict_nih.py [YOUR_IMAGE_PATH_HERE]
```
*Example:*
```bash
.\venv_gpu\Scripts\python.exe scripts/predict_nih.py test.png
```
