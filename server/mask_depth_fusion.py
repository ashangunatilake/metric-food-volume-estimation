import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ---------- LOAD MODELS ----------
# YOLO food segmentation
seg_model = YOLO("best.pt")

# MiDaS depth
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.dpt_transform

# ---------- LOAD IMAGE ----------
img_bgr = cv2.imread("test4.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]

# ---------- FOOD SEGMENTATION ----------
seg_results = seg_model(img_bgr, device=0)
mask = seg_results[0].masks.data[0].cpu().numpy()
mask = (mask > 0.5).astype(np.uint8)

# Resize mask to image size
mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

# ---------- DEPTH ESTIMATION ----------
input_tensor = transform(img_rgb)
with torch.no_grad():
    depth = midas(input_tensor)

depth = depth.squeeze().cpu().numpy()

# Resize depth to image size
depth = cv2.resize(depth, (w, h))

# ---------- MASK DEPTH ----------
masked_depth = depth * mask

print("Masked depth stats:")
print("Min:", masked_depth[mask > 0].min())
print("Max:", masked_depth[mask > 0].max())
print("Mean:", masked_depth[mask > 0].mean())

# ---------- VISUALIZATION ----------
depth_norm = cv2.normalize(masked_depth, None, 0, 255, cv2.NORM_MINMAX)
depth_norm = depth_norm.astype(np.uint8)

overlay = img_bgr.copy()
overlay[mask == 0] = 0

cv2.namedWindow("Food Mask", cv2.WINDOW_NORMAL) 
cv2.namedWindow("Masked Depth", cv2.WINDOW_NORMAL) 
cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)
cv2.imshow("Food Mask", mask * 255)
cv2.imshow("Masked Depth", depth_norm)
cv2.imshow("Overlay", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()
