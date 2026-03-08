import torch
import cv2
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
# Load MiDaS model
model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
model.eval()

# Load transforms
transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = transforms.dpt_transform

# Load image
img = cv2.imread("test3.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Run inference
input_tensor = transform(img)
with torch.no_grad():
    depth = model(input_tensor)

depth = depth.squeeze().cpu().numpy()

# Normalize for visualization
depth_norm = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
depth_norm = depth_norm.astype(np.uint8)

print("Depth stats:")
print("Min:", depth.min())
print("Max:", depth.max())
print("Mean:", depth.mean())


cv2.imshow("Depth Map", depth_norm)
cv2.waitKey(0)
cv2.destroyAllWindows()
