# check_dataset_model.py
import math
import torch
from torch.utils.data import DataLoader

from dataset import RotationDataset
from model import AngleNet
import torch.optim as optim


def angle_from_sincos(x):
    return math.degrees(math.atan2(x[0], x[1]))


def normalize_sincos(x):
    return x / (x.norm(dim=1, keepdim=True) + 1e-6)


def main():
    dataset = RotationDataset(
        image_dir="C:\\Users\\pasha\\isimagerotated\\data\\images",
        img_size=224,
        max_angle=15,
        augment=True,
        binarize=False,
        geom_aug=True,
        normalize=True,
    )

    loader = DataLoader(dataset, batch_size=16, shuffle=True)

    model = AngleNet(
        backbone="mobilenetv3_small_100",
        pretrained=True,
        in_chans=3,
    )

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.train()

    # ---- train 300 steps ----
    for step, batch in enumerate(loader):
        if step >= 300:
            break

        images = batch["image"]
        targets = batch["target"]

        preds = normalize_sincos(model(images))
        loss = 1.0 - (preds * targets).sum(dim=1).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"step {step:03d} | loss {loss.item():.4f}")

    # ---- eval ----
    model.eval()
    batch = next(iter(loader))
    images = batch["image"]
    true_angles = batch["angle_deg"]

    with torch.no_grad():
        preds = normalize_sincos(model(images))

    for i in range(len(images)):
        pred_angle = angle_from_sincos(preds[i].numpy())
        print(f"GT: {true_angles[i]:6.2f}°, " f"Pred: {pred_angle:6.2f}°")


if __name__ == "__main__":
    main()
