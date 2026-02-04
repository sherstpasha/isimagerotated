# dataset.py
import math
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class RotationDataset(Dataset):
    """
    Dataset for text rotation estimation.
    - self-supervised: random rotation
    - optional photo augmentation
    - optional binarization (recommended)
    """

    def __init__(
        self,
        image_dir,
        img_size=224,
        max_angle=15.0,
        augment=True,
        binarize=False,
        geom_aug=True,
        normalize=True,
    ):
        self.image_paths = sorted(
            [
                p
                for p in Path(image_dir).iterdir()
                if p.suffix.lower() in [".jpg", ".png", ".jpeg"]
            ]
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No images found in {image_dir}")

        self.img_size = img_size
        self.max_angle = float(max_angle)
        self.augment = augment
        self.binarize = binarize
        self.geom_aug = geom_aug
        self.normalize = normalize

        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __len__(self):
        return len(self.image_paths)

    # ---------------- augmentation ----------------

    def _random_affine(self, img, angle):
        h, w = img.shape[:2]

        scale = random.uniform(0.9, 1.1)
        tx = random.uniform(-0.05, 0.05) * w
        ty = random.uniform(-0.05, 0.05) * h

        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
        M[:, 2] += (tx, ty)

        return cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _rotate(self, img, angle):
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)

        return cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def _photo_aug(self, img):
        # blur
        if random.random() < 0.3:
            k = random.choice([3, 5])
            img = cv2.GaussianBlur(img, (k, k), 0)

        # brightness / contrast
        if random.random() < 0.5:
            alpha = random.uniform(0.8, 1.2)
            beta = random.uniform(-20, 20)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # gaussian noise
        if random.random() < 0.3:
            noise = np.random.normal(0, 5, img.shape).astype(np.float32)
            img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        return img

    # ---------------- binarization ----------------

    def _binarize(self, img):
        """
        Fast & robust binarization for text orientation.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        bw = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=31,
            C=15,
        )

        bw = cv2.medianBlur(bw, 3)

        return bw

    # ---------------- main ----------------

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        img = cv2.imread(str(img_path))
        if img is None:
            raise RuntimeError(f"Failed to read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))

        # --- sample angle ---
        angle = random.uniform(-self.max_angle, self.max_angle)

        # --- geometric aug ---
        if self.geom_aug:
            img = self._random_affine(img, angle)
        else:
            img = self._rotate(img, angle)

        # --- photo aug ---
        if self.augment:
            img = self._photo_aug(img)

        # --- binarization ---
        if self.binarize:
            bw = self._binarize(img)
            img = np.stack([bw, bw, bw], axis=-1)

        # --- to tensor ---
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)  # C,H,W

        if self.normalize:
            mean = self._mean[:, None, None]
            std = self._std[:, None, None]
            img = (img - mean) / std

        theta = math.radians(angle)
        target = np.array(
            [math.sin(theta), math.cos(theta)],
            dtype=np.float32,
        )

        return {
            "image": torch.from_numpy(img),
            "target": torch.from_numpy(target),
            "angle_deg": angle,
        }
