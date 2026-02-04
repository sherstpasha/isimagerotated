# model.py
import torch
import torch.nn as nn
import timm


class AngleNet(nn.Module):
    def __init__(
        self,
        backbone: str = "mobilenetv3_small_100",
        pretrained: bool = True,
        in_chans: int = 3,
        img_size: int = 224,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            in_chans=in_chans,
            num_classes=0,
            global_pool="avg",
        )

        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, img_size, img_size)
            feat = self.backbone(dummy)
            feat_dim = feat.shape[1]

        self.head = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        feats = self.backbone(x)
        out = self.head(feats)
        return out
