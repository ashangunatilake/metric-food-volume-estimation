import torch
import cv2
import numpy as np
from PIL import Image
import depth_pro

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
IMAGE_PATH = "IMG_3442.jpg"       # Change to your test image
OUTPUT_PATH = "depth_output.png"

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# LOAD MODEL
# -------------------------------------------------
print("Loading Depth Pro model...")
depth_model, depth_transform = depth_pro.create_model_and_transforms()
depth_model = depth_model.to(device)
depth_model.eval()
print("Depth Pro loaded.")

# -------------------------------------------------
# LOAD IMAGE
# -------------------------------------------------
img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_height, img_width = img_bgr.shape[:2]

img_pil = Image.fromarray(img_rgb)

# -------------------------------------------------
# PREPROCESS & INFER
# -------------------------------------------------
image_tensor = depth_transform(img_pil).unsqueeze(0).to(device)

print("Running Depth Pro inference...")
with torch.no_grad():
    prediction = depth_model.infer(image_tensor)   # focal length auto-estimated

depth_map = prediction["depth"].squeeze().cpu().numpy()   # shape: (H, W), metric depth in meters
focallength_px = prediction.get("focallength_px", None)

print(f"Depth map shape: {depth_map.shape}")
print(f"Estimated focal length (px): {focallength_px}")
print(f"Depth percentiles [1,25,50,75,99]: {np.percentile(depth_map, [1, 25, 50, 75, 99])}")

# -------------------------------------------------
# RESIZE TO ORIGINAL IMAGE SIZE
# -------------------------------------------------
depth_resized = cv2.resize(depth_map, (img_width, img_height), interpolation=cv2.INTER_CUBIC)

# -------------------------------------------------
# GRAYSCALE DEPTH MAP (near = bright, far = dark)
# -------------------------------------------------
depth_norm = cv2.normalize(depth_resized, None, 0, 255, cv2.NORM_MINMAX)
depth_uint8 = depth_norm.astype(np.uint8)
depth_gray = 255 - depth_uint8  # invert: closer objects appear brighter

# -------------------------------------------------
# SAVE OUTPUT
# -------------------------------------------------
cv2.imwrite(OUTPUT_PATH, depth_gray)
print(f"Saved grayscale depth map to: {OUTPUT_PATH}")

# Also save raw depth as 16-bit PNG (in millimeters) for downstream use
depth_mm = (depth_resized * 1000).astype(np.uint16)
cv2.imwrite("depth_raw_mm.png", depth_mm)
print("Saved raw depth map (mm, 16-bit) to: depth_raw_mm.png")

# -------------------------------------------------
# DISPLAY
# -------------------------------------------------
cv2.namedWindow("Depth Pro Output", cv2.WINDOW_NORMAL)
cv2.imshow("Depth Pro Output", depth_gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
