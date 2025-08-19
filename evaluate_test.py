print("[evaluate_test.py] 🔍 Starting script…")

import argparse
import torch
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# Must match your two test sub-folder names:
LABELS = ['FAKE', 'REAL']

def load_model(path, device):
    print(f"[DEBUG] Loading model from {path} onto {device}")
    # Construct the same architecture you trained
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(LABELS))
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model = model.to(device).eval()
    print("[DEBUG] Model loaded and ready")
    return model

def main(test_dir, model_path, batch_size):
    print(f"[DEBUG] test_dir = {test_dir}")
    print(f"[DEBUG] model_path = {model_path}")
    print(f"[DEBUG] batch_size = {batch_size}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(model_path, device)

    # Same preprocessing as during training
    tf = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],
                             [0.229,0.224,0.225])
    ])

    ds = datasets.ImageFolder(test_dir, transform=tf)
    print(f"[DEBUG] Found classes in test set: {ds.classes}")
    print(f"[DEBUG] Number of test images: {len(ds)}")

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    all_preds, all_labels = [], []
    correct, total = 0, 0

    with torch.no_grad():
        for batch_i, (imgs, labels) in enumerate(loader, 1):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs  = F.softmax(logits, dim=1)
            _, preds = torch.max(probs, 1)

            all_preds += preds.cpu().tolist()
            all_labels += labels.tolist()
            correct += (preds.cpu() == labels).sum().item()
            total   += labels.size(0)

            print(f"[DEBUG] Batch {batch_i}: Acc so far = {correct}/{total}")

    print("\n=== FINAL RESULTS ===")
    print(f"Overall Accuracy: {correct/total:.4f} ({correct}/{total})\n")
    print(classification_report(
        all_labels, all_preds,
        target_names=ds.classes, digits=4
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir",   default="data/test",
                        help="Root folder with FAKE/ and REAL/ subfolders")
    parser.add_argument("--model_path", default="image_detector.pth",
                        help="Path to your trained checkpoint")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for DataLoader")
    args = parser.parse_args()
    main(args.test_dir, args.model_path, args.batch_size)
