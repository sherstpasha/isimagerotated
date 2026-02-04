# train.py
import math
import time
import logging
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import RotationDataset
from model import AngleNet


# ---------------- utils ----------------


def setup_logger(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def angle_from_sincos(x):
    return math.degrees(math.atan2(x[0], x[1]))


def normalize_sincos(x):
    return x / (x.norm(dim=1, keepdim=True) + 1e-6)


def rotate_image(img, angle):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def draw_angle_vector(img, angle_deg, color, label):
    """
    Рисует вектор угла из центра изображения
    angle_deg — в градусах (как у тебя сейчас)
    """
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    length = min(h, w) // 3

    # ВАЖНО: угол у нас sin/cos => отсчёт от вертикали
    theta = math.radians(angle_deg)

    dx = int(length * math.sin(theta))
    dy = int(-length * math.cos(theta))  # минус, потому что y вниз

    end = (cx + dx, cy + dy)

    cv2.arrowedLine(
        img,
        (cx, cy),
        end,
        color,
        2,
        tipLength=0.15,
    )

    cv2.putText(
        img,
        f"{label}: {angle_deg:.1f}°",
        (10, 30 if label == "GT" else 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2,
        cv2.LINE_AA,
    )


def angle_from_sincos_human(x):
    # x = [sin, cos] — как в обучении
    angle_vert = math.degrees(math.atan2(x[0], x[1]))
    # переводим из "от вертикали" → "от горизонтали"
    angle_human = 90.0 - angle_vert
    return angle_human


def save_visual(batch, preds, save_path):
    images = batch["image"]
    gt_angles = batch["angle_deg"]

    B = min(4, len(images))
    rows = []

    for i in range(B):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = img * std + mean
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255).astype(np.uint8)

        gt_angle = float(gt_angles[i])
        pred_angle = angle_from_sincos(preds[i].cpu().numpy())

        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        # 1️⃣ GT corrected image (undo rotation)
        M_gt = cv2.getRotationMatrix2D(center, -gt_angle, 1.0)
        img_gt = cv2.warpAffine(
            img,
            M_gt,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        cv2.putText(
            img_gt,
            f"GT corrected: {-gt_angle:.1f} deg",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        # 2️⃣ input (already rotated)
        img_orig = img.copy()
        cv2.putText(
            img_orig,
            "Input (rotated)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # 3️⃣ corrected by predicted angle
        M_pred = cv2.getRotationMatrix2D(center, -pred_angle, 1.0)
        img_pred = cv2.warpAffine(
            img,
            M_pred,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        cv2.putText(
            img_pred,
            f"Pred angle: {pred_angle:.1f} deg",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

        row = np.hstack([img_gt, img_orig, img_pred])
        rows.append(row)

    canvas = np.vstack(rows)
    cv2.imwrite(str(save_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


# ---------------- curriculum ----------------


def get_max_angle(epoch: int) -> float:
    return 45.0


# ---------------- main ----------------


def main():
    # -------- config --------
    EXP_NAME = "exp_curriculum_512"
    IMAGE_DIR = r"C:\shared\Archives020525\train_images"

    EPOCHS = 80
    IMG_SIZE = 224
    BATCH_SIZE = 32  # ← ВАЖНО для 512
    LR = 3e-4  # ← стабильнее для большого разрешения
    VIS_EVERY = 5

    NUM_WORKERS = 6
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # -------- dirs --------
    exp_dir = Path("experiments") / EXP_NAME
    ckpt_dir = exp_dir / "checkpoints"
    vis_dir = exp_dir / "visuals"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    setup_logger(exp_dir / "train.log")

    logging.info(f"Experiment: {EXP_NAME}")
    logging.info(f"Device: {DEVICE}")

    # -------- data --------
    train_ds = RotationDataset(
        IMAGE_DIR,
        img_size=IMG_SIZE,
        max_angle=45.0,
        augment=True,
        binarize=False,
        geom_aug=True,
        normalize=True,
    )

    val_ds = RotationDataset(
        IMAGE_DIR,
        img_size=IMG_SIZE,
        max_angle=45.0,
        augment=False,
        binarize=False,
        geom_aug=False,
        normalize=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True,
        num_workers=NUM_WORKERS,
    )

    # -------- model --------
    model = AngleNet(
        backbone="mobilenetv3_small_100",
        pretrained=True,
        in_chans=3,
        img_size=IMG_SIZE,
    ).to(DEVICE)

    # ---- freeze backbone ----
    # for p in model.backbone.parameters():
    #     p.requires_grad = False

    # # ---- unfreeze last 2 blocks ----
    # if hasattr(model.backbone, "blocks"):
    #     for block in model.backbone.blocks[-2:]:
    #         for p in block.parameters():
    #             p.requires_grad = True
    # else:
    #     # fallback (на случай другого backbone)
    #     for p in list(model.backbone.parameters())[-20:]:
    #         p.requires_grad = True

    optimizer = optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_mae = 1e9

    # -------- training --------
    for epoch in range(EPOCHS):
        t0 = time.time()

        # ---- curriculum ----
        train_ds.max_angle = get_max_angle(epoch)
        logging.info(f"Epoch {epoch:03d} | max_angle = {train_ds.max_angle:.1f} deg")

        # ---- train ----
        model.train()
        train_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch:03d} [train]", leave=False):
            images = batch["image"].to(DEVICE, non_blocking=True)
            targets = batch["target"].to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
                preds = model(images)
                preds = normalize_sincos(preds)
                loss = 1.0 - (preds * targets).sum(dim=1).mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ---- val ----
        model.eval()
        maes = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch:03d} [val]", leave=False):
                images = batch["image"].to(DEVICE, non_blocking=True)
                gt_angles = batch["angle_deg"]

                preds = model(images)
                preds = normalize_sincos(preds)

                for i in range(len(images)):
                    pred_angle = angle_from_sincos(preds[i].cpu().numpy())
                    maes.append(abs(pred_angle - gt_angles[i]))

        val_mae = float(np.mean(maes))

        scheduler.step()

        # ---- logging ----
        logging.info(
            f"Epoch {epoch:03d} | "
            f"train_loss {train_loss:.4f} | "
            f"val_MAE {val_mae:.3f} deg | "
            f"lr {scheduler.get_last_lr()[0]:.2e} | "
            f"time {time.time() - t0:.1f}s"
        )

        # ---- save ----
        torch.save(model.state_dict(), ckpt_dir / "last.pt")

        if val_mae < best_mae:
            best_mae = val_mae
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            logging.info(f"New best MAE: {best_mae:.3f} deg")

        # ---- visualize ----
        if epoch % VIS_EVERY == 0:
            # train визуализация
            train_batch = next(iter(train_loader))
            train_images = train_batch["image"].to(DEVICE)

            with torch.no_grad():
                train_preds = model(train_images)
                train_preds = normalize_sincos(train_preds)

            save_visual(
                train_batch,
                train_preds,
                vis_dir / f"epoch_{epoch:03d}_train.png",
            )

            # val визуализация
            val_batch = next(iter(val_loader))
            val_images = val_batch["image"].to(DEVICE)

            with torch.no_grad():
                val_preds = model(val_images)
                val_preds = normalize_sincos(val_preds)

            save_visual(
                val_batch,
                val_preds,
                vis_dir / f"epoch_{epoch:03d}_val.png",
            )


if __name__ == "__main__":
    main()
