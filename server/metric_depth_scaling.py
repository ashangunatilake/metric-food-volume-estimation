import numpy as np
import cv2
import torch
from ultralytics import YOLO

# ---------- SIMULATED ARKIT INPUT ----------
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

# ---------- SEGMENT FOOD ----------
seg_results = seg_model(img_bgr, device=0)
food_mask = seg_results[0].masks.data[0].cpu().numpy()
food_mask = (food_mask > 0.5).astype(np.uint8)
food_mask = cv2.resize(food_mask, (w, h), interpolation=cv2.INTER_NEAREST)

# ---------- DEPTH ESTIMATION ----------
input_tensor = transform(img_rgb)
with torch.no_grad():
    depth_rel = midas(input_tensor)

depth_rel = depth_rel.squeeze().cpu().numpy()
depth_rel = cv2.resize(depth_rel, (w, h))

# ---------- ESTIMATE PLANE RELATIVE DEPTH ----------
# Assume plane = background (mask == 0)
plane_depth_rel = depth_rel[food_mask == 0].mean()

# ---------- SCALE RECOVERY ----------
scale = plane_distance_m / plane_depth_rel
depth_metric = depth_rel * scale

# ---------- FOOD METRIC DEPTH ----------
food_depth_m = depth_metric * food_mask

print("Metric depth stats (food):")
print("Min (m):", food_depth_m[food_mask > 0].min())
print("Max (m):", food_depth_m[food_mask > 0].max())
print("Mean (m):", food_depth_m[food_mask > 0].mean())

# ---------- VISUALIZATION ----------

depth_vis = cv2.normalize(food_depth_m, None, 0, 255, cv2.NORM_MINMAX)
depth_vis = depth_vis.astype(np.uint8)

cv2.namedWindow("Food Metric Depth", cv2.WINDOW_NORMAL) 
cv2.imshow("Food Metric Depth", depth_vis)
cv2.waitKey(0)
cv2.destroyAllWindows()
