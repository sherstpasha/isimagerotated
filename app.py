# app.py
import math
import cv2
import numpy as np
import torch
import gradio as gr
from PIL import Image

from anglenet.model import AngleNet


# ---------------- config ----------------
CHECKPOINT_PATH = "experiments\\exp_curriculum_512\\checkpoints\\best.pt"
IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BACKBONE = "mobilenetv3_small_100"
PRETRAINED = False  # веса берутся из чекпоинта, чтобы не скачивать backbone
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[:, None, None]
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[:, None, None]


# ---------------- utils ----------------
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


def preprocess(img):
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = (img - MEAN) / STD
    return torch.from_numpy(img).unsqueeze(0)


# ---------------- load model ----------------
model = AngleNet(
    backbone=BACKBONE,
    pretrained=PRETRAINED,
    in_chans=3,
    img_size=IMG_SIZE,
).to(DEVICE)

state = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()


# ---------------- inference ----------------
def correct_rotation(image: Image.Image, steps: int):
    if image is None:
        return None, ""

    # PIL -> numpy RGB
    img = np.array(image.convert("RGB"))
    current = img
    step_angles = []

    for _ in range(int(steps)):
        inp = preprocess(current).to(DEVICE)

        with torch.inference_mode():
            pred = model(inp)
            pred = normalize_sincos(pred)[0].cpu().numpy()

        angle = angle_from_sincos(pred)
        step_angles.append(angle)
        current = rotate_image(current, -angle)

    # numpy -> PIL
    corrected = Image.fromarray(current)

    total = float(np.sum(step_angles)) if step_angles else 0.0
    lines = [f"step {i+1}: {a:.2f} deg" for i, a in enumerate(step_angles)]
    lines.append(f"total: {total:.2f} deg")

    return corrected, "\n".join(lines)


# ---------------- gradio interface ----------------
iface = gr.Interface(
    fn=correct_rotation,
    inputs=[
        gr.Image(
            sources=["upload", "webcam"],
            type="pil",  # ← КЛЮЧЕВО
            label="Исходное изображение",
        ),
        gr.Slider(
            minimum=1,
            maximum=3,
            step=1,
            value=1,
            label="Количество шагов",
        ),
    ],
    outputs=[
        gr.Image(type="pil", label="Скорректированное изображение"),
        gr.Textbox(label="Предсказанный угол (градусы)"),
    ],
    title="📐 Text Rotation Auto-Correction",
    description="Сфотографируй или загрузи изображение с текстом — модель автоматически выровняет его.",
)


if __name__ == "__main__":
    iface.launch(share=True)
