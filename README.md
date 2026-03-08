# Metric Food Volume Estimation

Estimates the volume of food in an image using ARKit plane detection (iOS), monocular depth estimation (Apple Depth Pro), and food instance segmentation (YOLOv8).

```
iOS Client (ARKit)  ──POST /estimate_volume──►  FastAPI Server
   - captures frame                              - food segmentation (YOLOv8)
   - detects horizontal plane                    - metric depth (Depth Pro)
   - sends image + camera intrinsics             - volume integration
   - displays cm³ result         ◄──────────────
```

> **Model weights are not tracked in git.** Download them separately (see below).

---

## Server Setup

### Prerequisites

- Python 3.10+
- ~5 GB free disk space for models
- A CPU is sufficient; CUDA is used automatically if available

### 1. Clone and enter the server directory

```bash
cd server
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

This also installs Apple Depth Pro directly from its GitHub repository.

### 4. Download model weights

#### Depth Pro checkpoint

Download `depth_pro.pt` from the [ml-depth-pro releases] and place it at:

```
server/checkpoints/depth_pro.pt
```

You can download the weights using:

```
wget https://ml-site.cdn-apple.com/models/depth-pro/depth_pro.pt -P checkpoints
```

#### Food segmentation model

Place your trained YOLOv8 segmentation weights at:

```
server/models/best_foodseg.pt
```

### 5. Run the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

The server will be available at `http://<your-machine-ip>:8000`.  
Make sure your machine and iPhone are on the **same Wi-Fi network**.

---

## iOS Client Setup

### Prerequisites

- macOS with Xcode 15+
- Flutter SDK 3.x (`flutter --version` to verify)
- A **physical iPhone** with iOS 14+ — ARKit does not run on the Simulator
- Apple Developer account (free tier is sufficient for local deployment)

### 1. Get Flutter dependencies

```bash
cd client
flutter pub get
```

### 2. Install CocoaPods dependencies

```bash
cd ios
pod install
cd ..
```

### 3. Open in Xcode

Open the workspace (not the project file):

```bash
open ios/Runner.xcworkspace
```

### 4. Configure signing

1. In Xcode, select the **Runner** target → **Signing & Capabilities**
2. Set your **Team** (Apple ID)
3. Change the **Bundle Identifier** to something unique, e.g. `com.yourname.foodvolume`

### 5. ARKit capability

ARKit is already implemented natively in `ios/Runner/AppDelegate.swift`. The app:

- Registers an `arkit_view` platform view that renders `ARSCNView`
- Runs `ARWorldTrackingConfiguration` with horizontal plane detection
- On "Capture + Send", captures the current `ARFrame` and returns:
  - JPEG image bytes
  - Camera intrinsics (`fx`, `fy`, `cx`, `cy`)
  - Closest detected horizontal plane distance, center, and normal

No additional Xcode capability toggles are required — ARKit access is granted automatically when `ARSCNView` is used. If you add features that require the camera privacy string, add `NSCameraUsageDescription` to `ios/Runner/Info.plist`.

### 6. Set the backend endpoint

With the app running on device, update the **Endpoint** field to:

```
http://<server-ip>:8000/estimate_volume
```

Replace `<server-ip>` with the LAN IP of your server machine (e.g. `192.168.1.4`). The default in `lib/main.dart` is `http://192.168.1.4:8000/estimate_volume` — update `kDefaultEndpoint` if you want a different default.

### 7. Build and run

Select your connected iPhone as the target in Xcode and press **Run**, or:

```bash
flutter run --release
```

---

## Project Structure

```
server/
  main.py                   # FastAPI app — /estimate_volume endpoint
  food_volume_estimation.py # Standalone volume estimation script
  mask_depth_fusion.py      # Depth + mask fusion utilities
  metric_depth_scaling.py   # ARKit-based metric depth scaling
  requirements.txt
  checkpoints/
    depth_pro.pt            # ⚠ not in git — download separately
  models/
    best_foodseg.pt         # ⚠ not in git — download separately

client/
  lib/main.dart             # Flutter app — ARKit capture + HTTP upload
  ios/Runner/
    AppDelegate.swift       # ARKit platform view + MethodChannel bridge
```
