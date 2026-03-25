from pathlib import Path
import torch
from torchvision.models import ResNet18_Weights, resnet18

base_dir = Path(__file__).resolve().parent
out_path = base_dir / "pretrained_weights" / "resnet18_imagenet.pth"
out_path.parent.mkdir(parents=True, exist_ok=True)

model = resnet18(weights=ResNet18_Weights.DEFAULT)
torch.save(model.state_dict(), out_path)

print(f"Saved pretrained weights to: {out_path}")