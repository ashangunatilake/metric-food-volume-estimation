from ultralytics import YOLO
import cv2

# Load YOLOv8 instance segmentation model
model = YOLO("models/best_foodseg.pt")  # high-accuracy segmentation model

# Load image
img = cv2.imread("IMG_3442.jpg")

# Run inference on GPU
results = model(img, device=0)

# Visualize results
annotated = results[0].plot()

# Save the segmented output
cv2.imwrite("segmented_output.jpg", annotated)
print("Saved segmented output to segmented_output.jpg")

cv2.namedWindow("YOLOv8 Segmentation", cv2.WINDOW_NORMAL) 
cv2.imshow("YOLOv8 Segmentation", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()