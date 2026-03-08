from fastapi import FastAPI, UploadFile, File, Form
import numpy as np
import cv2
import torch
from ultralytics import YOLO
import depth_pro
from PIL import Image
import json

# -------------------------------------------------
# DEVICE
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# APP
# -------------------------------------------------
app = FastAPI()

# -------------------------------------------------
# LOAD MODELS
# -------------------------------------------------
seg_model = YOLO("models/best_foodseg.pt")

depth_model, depth_transform = depth_pro.create_model_and_transforms()
depth_model = depth_model.to(device)
depth_model.eval()

print("Depth Pro loaded.")

# ROOT
@app.get("/")
def root():
    return {"status": "Food volume backend (Depth Pro - Clean Geometric Version)"}

# VOLUME ENDPOINT
@app.post("/estimate_volume")
async def estimate_volume(
    image: UploadFile = File(...),
    plane_center: str = Form(...),
    plane_normal: str = Form(...),
    plane_distance_m: float = Form(...),
    fx: float = Form(...),
    fy: float = Form(...),
    cx: float = Form(...),
    cy: float = Form(...)
):

    print("\n" + "="*60)
    print("INCOMING REQUEST")
    print("="*60)
    print("ARKit plane distance:", plane_distance_m)
    print("="*60)

    # Parse ARKit Plane (not used)
    plane_center = np.array(json.loads(plane_center))
    plane_normal = np.array(json.loads(plane_normal))
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Read Image
    contents = await image.read()
    np_img = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    img_height, img_width = img_bgr.shape[:2]

    # SEGMENTATION
    seg_results = seg_model(img_bgr)

    if seg_results[0].masks is None:
        return {"error": "No food detected"}

    masks = seg_results[0].masks.data.cpu().numpy()
    class_ids = seg_results[0].boxes.cls.cpu().numpy().astype(int)
    class_names = seg_results[0].names

    # DEPTH PRO INFERENCE
    img_pil = Image.fromarray(img_rgb)
    image_tensor = depth_transform(img_pil).unsqueeze(0).to(device)

    _, _, H_model, W_model = image_tensor.shape

    scale_x = W_model / img_width
    f_px_scaled = fx * scale_x
    f_px_tensor = torch.tensor([f_px_scaled], dtype=torch.float32).to(device)

    with torch.no_grad():
        prediction = depth_model.infer(image_tensor, f_px=f_px_tensor)

    depth_metric = prediction["depth"].squeeze().cpu().numpy()

    depth_metric = cv2.resize(
        depth_metric,
        (img_width, img_height),
        interpolation=cv2.INTER_CUBIC
    )

    print("Depth raw percentiles:",
          np.percentile(depth_metric, [1, 25, 50, 75, 99]))

    # TABLE MASK
    combined_mask = np.zeros((img_height, img_width), dtype=np.uint8)

    for mask in masks:
        mask_resized = cv2.resize(
            (mask > 0.5).astype(np.uint8),
            (img_width, img_height),
            interpolation=cv2.INTER_NEAREST
        )
        combined_mask |= mask_resized

    table_mask = combined_mask == 0

    table_depth_values = depth_metric[table_mask]

    if len(table_depth_values) < 100:
        return {"error": "Insufficient table region"}

    # METRIC SCALE USING ARKIT PLANE DISTANCE
    table_depth_median = np.median(table_depth_values)
    scale_factor = plane_distance_m / table_depth_median

    print("Table depth median (model space):", table_depth_median)
    print("Scale factor:", scale_factor)

    depth_metric_scaled = depth_metric * scale_factor

    # HEIGHT RELATIVE TO TABLE MEDIAN
    table_depth_scaled_median = np.median(
        depth_metric_scaled[table_mask]
    )

    height = table_depth_scaled_median - depth_metric_scaled
    height = np.maximum(height, 0)

    print("Height percentiles:",
          np.percentile(height[combined_mask == 1],
                        [50, 75, 90, 95, 99]))

    # CORRECT PER-PIXEL AREA (Perspective Correct)
    pixel_area = (depth_metric_scaled ** 2) / (fx * fy)

    # VOLUME INTEGRATION
    kernel = np.ones((3, 3), np.uint8)

    class_volumes = {}
    total_volume_m3 = 0.0

    for mask, class_id in zip(masks, class_ids):

        class_name = class_names[class_id]

        mask_resized = cv2.resize(
            (mask > 0.5).astype(np.uint8),
            (img_width, img_height),
            interpolation=cv2.INTER_NEAREST
        )

        mask_refined = cv2.erode(mask_resized, kernel, iterations=1)

        volume_m3 = np.sum(height * pixel_area * mask_refined)

        class_volumes[class_name] = volume_m3
        total_volume_m3 += volume_m3

        print(f"{class_name} volume:",
              volume_m3 * 1e6, "cm³")

    total_volume_cm3 = total_volume_m3 * 1e6

    print("TOTAL VOLUME:", total_volume_cm3, "cm³")

    return {
        "volume_cm3": float(total_volume_cm3),
        "volume_per_class_cm3": {
            k: float(v * 1e6) for k, v in class_volumes.items()
        },
        "scale_factor_used": float(scale_factor),
        "plane_distance_m": plane_distance_m
    }