# image_detector/image_detection.py
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 1) Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2) Build the same ResNet50 + 2-class head you trained
def _build_model():
    model = models.resnet50(pretrained=False)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 2)
    return model

# 3) Instantiate & load your fine-tuned weights
_MODEL = _build_model().to(DEVICE)

# Assume you placed your .pth next to this file, adjust if not:
MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "image_detector.pth")
)

_state = torch.load(MODEL_PATH, map_location=DEVICE)
_MODEL.load_state_dict(_state)
_MODEL.eval()

# 4) Your validation transforms (must match training *val* pipeline)
_TRANSFORMS = transforms.Compose([
    transforms.Resize(int(224 * 1.15)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def detect_ai_image(image_path: str) -> dict:
    """
    Args:
      image_path: filesystem path to JPG/PNG

    Returns:
      {
        "label":       "AI-Generated" or "Real",
        "confidence":  float probability of that class (0–1)
      }
    """
    # 5) Load & preprocess
    img = Image.open(image_path).convert("RGB")
    x   = _TRANSFORMS(img).unsqueeze(0).to(DEVICE)

    # 6) Forward pass
    with torch.no_grad():
        logits = _MODEL(x)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()

    # 7) Map indices → your folders:
    #    During training you likely had `ai_generated` as class 0, `real` as class 1.
    ai_prob, real_prob = float(probs[0]), float(probs[1])

    if ai_prob > real_prob:
        return {"label": "AI-Generated", "confidence": ai_prob}
    else:
        return {"label": "Real", "confidence": real_prob}
