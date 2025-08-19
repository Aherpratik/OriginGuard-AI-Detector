
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def get_data_loaders(data_dir, batch_size=32, image_size=224):
    # Standard ImageNet normalization
    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(image_size*1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        normalize,
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform=train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, 'val'),   transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, train_ds.classes

def build_model(num_classes=2, feature_extract=True):
    model = models.resnet50(pretrained=True)
    # Freeze feature extractor if desired
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
    # Replace final fully‐connected layer
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

def train(model, dataloaders, device, epochs, lr, output_path):
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_acc = 0.0

    for epoch in range(1, epochs+1):
        print(f"Epoch {epoch}/{epochs}")
        for phase in ['train', 'val']:
            loader = dataloaders[phase]
            model.train(phase=='train')
            running_loss, running_corrects, total = 0.0, 0, 0

            for inputs, labels in tqdm(loader, desc=phase):
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase=='train':
                        loss.backward()
                        optimizer.step()

                running_loss   += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds==labels.data)
                total += inputs.size(0)

            epoch_loss = running_loss / total
            epoch_acc  = running_corrects.double() / total
            print(f"  {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # Save best
            if phase=='val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), output_path)
                print(f"  → New best model saved ({best_acc:.4f})")

    print(f"Training complete. Best val Acc: {best_acc:.4f}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune ResNet50 for AI vs Real detection")
    parser.add_argument('--data_dir', required=True, help="Root data directory with train/ and val/ subfolders")
    parser.add_argument('--output_model', default='image_detector.pth', help="Where to save the best model")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--no_freeze', action='store_true',
                        help="Unfreeze all layers (fine-tune entire network)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)

    train_loader, val_loader, classes = get_data_loaders(
        args.data_dir, batch_size=args.batch_size)
    model = build_model(num_classes=len(classes),
                        feature_extract=not args.no_freeze).to(device)

    dataloaders = {'train': train_loader, 'val': val_loader}
    train(model, dataloaders, device,
          epochs=args.epochs, lr=args.lr,
          output_path=args.output_model)

if __name__ == '__main__':
    main()
