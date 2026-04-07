import torch
import torch.nn as nn
from pathlib import Path

try:
    from torchvision.models import resnet18
except Exception as exc:
    raise ImportError(
        "torchvision with ResNet18 support is required for this experiment. "
        "Please ensure torchvision is installed in the active environment."
    ) from exc

class ResNet18Transfer(nn.Module):
    def __init__(self, n_classes: int, mode: str = "finetuned_resnet18") -> None:
        super().__init__()

        if mode not in {"frozen_resnet18", "finetuned_resnet18"}:
            raise ValueError(f"Unsupported mode '{mode}'")

        weights_path = Path(__file__).resolve().parent.parent / "pretrained_weights" / "resnet18_imagenet.pth"

        if not weights_path.is_file():
            weights_path = Path(__file__).resolve().parent / "pretrained_weights" / "resnet18_imagenet.pth"
            if not weights_path.is_file():
                raise FileNotFoundError(
                    f"Missing local pretrained weights file: {weights_path}\n"
                    "Expected local ResNet18 ImageNet weights for submission-safe transfer learning."
                )

        backbone = resnet18(weights=None)
        state_dict = torch.load(weights_path, map_location="cpu")
        backbone.load_state_dict(state_dict)

        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        backbone.conv1 = new_conv
        backbone.fc = nn.Linear(backbone.fc.in_features, n_classes)

        self.model = backbone
        self.mode = mode

        self.register_buffer("norm_mean", torch.tensor([0.449], dtype=torch.float32).view(1, 1, 1, 1))
        self.register_buffer("norm_std", torch.tensor([0.226], dtype=torch.float32).view(1, 1, 1, 1))

        if mode == "frozen_resnet18":
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.fc.parameters():
                param.requires_grad = True
        else:
            for param in self.model.parameters():
                param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - self.norm_mean) / self.norm_std
        return self.model(x)