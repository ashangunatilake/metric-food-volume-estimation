import numpy as np
import cv2
import torch
from ultralytics import YOLO

# ---------- ARKIT INPUT ----------
plane_distance_m = 0.75   # meters
fx = 1200                 # pixels
fy = 1200                 # pixels

# ---------- LOAD MODELS ----------
seg_model = YOLO("best.pt")

midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
midas.eval()

transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.dpt_transform

# ---------- LOAD IMAGE ----------
img_bgr = cv2.imread("test5.jpg")
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
h, w = img_rgb.shape[:2]

# ---------- FOOD SEGMENTATION ----------
seg_results = seg_model(img_bgr, device=0)
mask = seg_results[0].masks.data[0].cpu().numpy()
mask = (mask > 0.5).astype(np.uint8)
mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

# ---------- DEPTH ESTIMATION ----------
input_tensor = transform(img_rgb)
with torch.no_grad():
    depth_rel = midas(input_tensor)

depth_rel = depth_rel.squeeze().cpu().numpy()
depth_rel = cv2.resize(depth_rel, (w, h))

# ---------- SCALE RECOVERY ----------
plane_depth_rel = depth_rel[mask == 0].mean()
scale = plane_distance_m / plane_depth_rel
depth_m = depth_rel * scale

# ---------- HEIGHT ABOVE PLANE ----------
height_map = plane_distance_m - depth_m
height_map[height_map < 0] = 0
height_map *= mask

# ---------- PER-PIXEL AREA ----------
pixel_area = (depth_m ** 2) / (fx * fy)

# ---------- VOLUME INTEGRATION ----------
volume_m3 = np.sum(pixel_area * height_map)
volume_cm3 = volume_m3 * 1e6

print(f"Estimated Food Volume: {volume_cm3:.2f} cm³")
