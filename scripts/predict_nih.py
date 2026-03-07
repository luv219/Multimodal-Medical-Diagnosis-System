import os
import sys
import torch
import argparse
from torchvision import transforms
from PIL import Image

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.image_only_model import ImageOnlyModel

def predict(image_path, model_path="results/best_model.pth"):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    NUM_CLASSES = 5
    LABEL_NAMES = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion"]

    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model weights not found at {model_path}. Please train the model first.")
        return

    # Load Model
    print(f"Loading model on {DEVICE}...")
    model = ImageOnlyModel(num_classes=NUM_CLASSES, use_pretrained=False).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()

    # Preprocess Image
    # Must match the val_transform used during training exactly
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(DEVICE) # Add batch dimension
    except Exception as e:
        print(f"Failed to load and transform image: {e}")
        return

    # Predict
    print(f"\nAnalyzing '{image_path}'...")
    with torch.no_grad(), torch.amp.autocast('cuda' if DEVICE == "cuda" else 'cpu'):
        logits = model(img_tensor)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()

    print("-" * 30)
    print(f"{'Pathology':<15} | {'Risk / Confidence'}")
    print("-" * 30)
    
    report_lines = [
        "========================================",
        f"| Inference Report for {os.path.basename(image_path)}",
        "========================================",
        ""
    ]
    
    for i, name in enumerate(LABEL_NAMES):
        prob_str = f"{probs[i]*100:.2f}%"
        print(f"{name:<15} | {prob_str}")
        report_lines.append(f"{name:<20} | {prob_str}")
        
    print("-" * 30)

    # Exporting Results to mimic the original application's `results/` structure
    import matplotlib.pyplot as plt
    
    file_prefix = os.path.splitext(os.path.basename(image_path))[0]
    out_dir = os.path.join(os.path.dirname(model_path), f"{file_prefix}_inference")
    os.makedirs(out_dir, exist_ok=True)
    
    # Dump Text Report (mimics 100epoch_imageOnly.txt)
    txt_path = os.path.join(out_dir, f"{file_prefix}_report.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(report_lines))
        
    # Dump Bar Chart Plot (mimics acc.png/ROC plots)
    plt.figure(figsize=(10, 6))
    bars = plt.bar(LABEL_NAMES, probs * 100, color='skyblue')
    plt.ylabel('Confidence (%)')
    plt.title(f'Diagnostic Confidence: {os.path.basename(image_path)}')
    plt.ylim(0, 100)
    
    # Add percentage labels above bars
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval:.1f}%', ha='center', va='bottom')
        
    plot_path = os.path.join(out_dir, f"{file_prefix}_plot.png")
    plt.savefig(plot_path)
    plt.close()
    
    print(f"\nSaved inference report and plot to: {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict diseases from a chest X-ray image.")
    parser.add_argument("image_path", type=str, help="Path to the chest X-ray image (.png/.jpg)")
    parser.add_argument("--model", type=str, default="results/best_model.pth", help="Path to the saved model weights")
    args = parser.parse_args()

    predict(args.image_path, args.model)
