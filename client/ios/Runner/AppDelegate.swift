import ARKit
import Flutter
import SceneKit
import UIKit

final class ARKitViewRegistry {
  static let shared = ARKitViewRegistry()
  weak var currentView: ARKitPlatformView?
}

final class ARKitViewFactory: NSObject, FlutterPlatformViewFactory {
  private let messenger: FlutterBinaryMessenger

  init(messenger: FlutterBinaryMessenger) {
    self.messenger = messenger
    super.init()
  }

  func create(
    withFrame frame: CGRect,
    viewIdentifier viewId: Int64,
    arguments args: Any?
  ) -> FlutterPlatformView {
    return ARKitPlatformView(frame: frame, viewIdentifier: viewId, arguments: args)
  }
}

final class ARKitPlatformView: NSObject, FlutterPlatformView, ARSessionDelegate {
  private struct ClosestPlaneData {
    let distanceMeters: Double
    let center: SIMD3<Float>
    let normal: SIMD3<Float>
  }

  private let sceneView: ARSCNView
  private var hasStartedSession = false
  private var planeNodes: [UUID: SCNNode] = [:]

  init(frame: CGRect, viewIdentifier viewId: Int64, arguments args: Any?) {
    sceneView = ARSCNView(frame: frame)
    super.init()
    sceneView.session.delegate = self
    sceneView.automaticallyUpdatesLighting = true
    sceneView.scene = SCNScene()
    ARKitViewRegistry.shared.currentView = self
    startSessionIfNeeded()
  }

  func view() -> UIView {
    return sceneView
  }

  func startSessionIfNeeded() {
    guard !hasStartedSession else { return }
    hasStartedSession = true
    let configuration = ARWorldTrackingConfiguration()
    configuration.planeDetection = [.horizontal]
    configuration.environmentTexturing = .automatic
    sceneView.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
  }

  func captureFrame(result: @escaping FlutterResult) {
    guard let frame = sceneView.session.currentFrame else {
      result(FlutterError(code: "NO_FRAME", message: "ARFrame not available yet.", details: nil))
      return
    }

    guard let planeData = closestPlaneData(frame: frame) else {
      result(FlutterError(code: "NO_PLANE", message: "No horizontal plane detected.", details: nil))
      return
    }

    let intrinsics = frame.camera.intrinsics
    let fx = Double(intrinsics.columns.0.x)
    let fy = Double(intrinsics.columns.1.y)
    let cx = Double(intrinsics.columns.2.x)
    let cy = Double(intrinsics.columns.2.y)

    guard let jpegData = jpegDataFrom(pixelBuffer: frame.capturedImage) else {
      result(FlutterError(code: "IMAGE_FAIL", message: "Failed to encode image.", details: nil))
      return
    }

    let payload: [String: Any] = [
      "fx": fx,
      "fy": fy,
      "cx": cx,
      "cy": cy,
      "plane_center": [
        Double(planeData.center.x),
        Double(planeData.center.y),
        Double(planeData.center.z)
      ],
      "plane_normal": [
        Double(planeData.normal.x),
        Double(planeData.normal.y),
        Double(planeData.normal.z)
      ],
      "plane_distance_m": planeData.distanceMeters,
      "image_bytes": FlutterStandardTypedData(bytes: jpegData)
    ]
    result(payload)
  }

  private func closestPlaneData(frame: ARFrame) -> ClosestPlaneData? {
    let cameraTransform = frame.camera.transform
    
    // Camera position in WORLD coordinates
    let cameraPosition = SIMD3<Float>(
        cameraTransform.columns.3.x,
        cameraTransform.columns.3.y,
        cameraTransform.columns.3.z
    )
    
    // World → Camera transform
    let worldToCamera = simd_inverse(cameraTransform)
    
    var best: ClosestPlaneData?
    var bestDistance: Float = .infinity
    
    for anchor in frame.anchors {
        guard let plane = anchor as? ARPlaneAnchor,
              plane.alignment == .horizontal else { continue }
        
        // ------------------------------------------------
        // 1️⃣ Plane center in WORLD coordinates
        // ------------------------------------------------
        let localCenter = SIMD4<Float>(plane.center.x, 0, plane.center.z, 1)
        let worldCenter4 = simd_mul(plane.transform, localCenter)
        let planePointWorld = SIMD3<Float>(
            worldCenter4.x,
            worldCenter4.y,
            worldCenter4.z
        )
        
        // ------------------------------------------------
        // 2️⃣ Plane normal in WORLD coordinates
        // ------------------------------------------------
        let planeNormalWorld = simd_normalize(
            SIMD3<Float>(
                plane.transform.columns.1.x,
                plane.transform.columns.1.y,
                plane.transform.columns.1.z
            )
        )
        
        // ------------------------------------------------
        // 3️⃣ Convert plane center → CAMERA coordinates
        // ------------------------------------------------
        let planePointWorld4 = SIMD4<Float>(
            planePointWorld.x,
            planePointWorld.y,
            planePointWorld.z,
            1.0
        )
        let planePointCamera4 = simd_mul(worldToCamera, planePointWorld4)
        let planePointCamera = SIMD3<Float>(
            planePointCamera4.x,
            planePointCamera4.y,
            planePointCamera4.z
        )
        
        // ------------------------------------------------
        // 4️⃣ Convert plane normal → CAMERA coordinates
        // ------------------------------------------------
        let planeNormalWorld4 = SIMD4<Float>(
            planeNormalWorld.x,
            planeNormalWorld.y,
            planeNormalWorld.z,
            0.0
        )
        let planeNormalCamera4 = simd_mul(worldToCamera, planeNormalWorld4)
        let planeNormalCamera = simd_normalize(
            SIMD3<Float>(
                planeNormalCamera4.x,
                planeNormalCamera4.y,
                planeNormalCamera4.z
            )
        )
        
        // ------------------------------------------------
        // 5️⃣ Compute viewing-direction distance
        // ------------------------------------------------
        let cameraForward = -SIMD3<Float>(
            cameraTransform.columns.2.x,
            cameraTransform.columns.2.y,
            cameraTransform.columns.2.z
        )
        
        let heightDiff = cameraPosition.y - planePointWorld.y
        let cosAngle = abs(simd_dot(cameraForward, planeNormalWorld))
        
        guard cosAngle > 0.1 else { continue }
        
        let distance = abs(heightDiff) / cosAngle
        
        if distance < bestDistance && distance > 0.01 && distance < 2.0 {
            bestDistance = distance
            
            // ================= DEBUG BLOCK =================
            NSLog("========== ARKIT DEBUG ==========")
            NSLog("Camera position (world): %@", String(describing: cameraPosition))
            NSLog("Camera forward (world): %@", String(describing: cameraForward))
            
            NSLog("Plane center (world): %@", String(describing: planePointWorld))
            NSLog("Plane normal (world): %@", String(describing: planeNormalWorld))
            
            NSLog("Plane center (camera): %@", String(describing: planePointCamera))
            NSLog("Plane normal (camera): %@", String(describing: planeNormalCamera))
            
            NSLog("Height difference: %f", heightDiff)
            NSLog("Cos(angle): %f", cosAngle)
            NSLog("Computed plane distance (m): %f", distance)
            NSLog("=================================")
            // =================================================
            
            best = ClosestPlaneData(
                distanceMeters: Double(distance),
                center: planePointCamera,   // CAMERA COORDS
                normal: planeNormalCamera  // CAMERA COORDS
            )
        }
    }
    
    return best
  }

  private func jpegDataFrom(pixelBuffer: CVPixelBuffer) -> Data? {
    let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
    let context = CIContext(options: nil)
    guard let cgImage = context.createCGImage(ciImage, from: ciImage.extent) else {
      return nil
    }
    let image = UIImage(cgImage: cgImage)
    return image.jpegData(compressionQuality: 0.9)
  }

  func session(_ session: ARSession, didAdd anchors: [ARAnchor]) {
    for anchor in anchors {
      guard let plane = anchor as? ARPlaneAnchor, plane.alignment == .horizontal else { continue }
      let node = makePlaneNode(for: plane)
      sceneView.scene.rootNode.addChildNode(node)
      planeNodes[plane.identifier] = node
    }
  }

  func session(_ session: ARSession, didUpdate anchors: [ARAnchor]) {
    for anchor in anchors {
      guard let plane = anchor as? ARPlaneAnchor, plane.alignment == .horizontal else { continue }
      if let node = planeNodes[plane.identifier],
         let planeGeometry = node.geometry as? SCNPlane {
        planeGeometry.width = CGFloat(plane.extent.x)
        planeGeometry.height = CGFloat(plane.extent.z)
        node.position = SCNVector3(plane.center.x, 0, plane.center.z)
      } else {
        let node = makePlaneNode(for: plane)
        sceneView.scene.rootNode.addChildNode(node)
        planeNodes[plane.identifier] = node
      }
    }
  }

  func session(_ session: ARSession, didRemove anchors: [ARAnchor]) {
    for anchor in anchors {
      guard let plane = anchor as? ARPlaneAnchor else { continue }
      if let node = planeNodes.removeValue(forKey: plane.identifier) {
        node.removeFromParentNode()
      }
    }
  }

  private func makePlaneNode(for plane: ARPlaneAnchor) -> SCNNode {
    let geometry = SCNPlane(width: CGFloat(plane.extent.x), height: CGFloat(plane.extent.z))
    geometry.firstMaterial?.diffuse.contents = UIColor.systemTeal.withAlphaComponent(0.3)
    geometry.firstMaterial?.isDoubleSided = true
    let node = SCNNode(geometry: geometry)
    node.position = SCNVector3(plane.center.x, 0, plane.center.z)
    node.eulerAngles.x = -.pi / 2
    return node
  }
}

@main
@objc class AppDelegate: FlutterAppDelegate {
  override func application(
    _ application: UIApplication,
    didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
  ) -> Bool {
    GeneratedPluginRegistrant.register(with: self)
    if let controller = window?.rootViewController as? FlutterViewController {
      let factory = ARKitViewFactory(messenger: controller.binaryMessenger)
      let registrar = self.registrar(forPlugin: "ARKitView")
      registrar?.register(factory, withId: "arkit_view")

      let channel = FlutterMethodChannel(
        name: "arkit_capture",
        binaryMessenger: controller.binaryMessenger
      )
      channel.setMethodCallHandler { call, result in
        switch call.method {
        case "captureFrame":
          guard let view = ARKitViewRegistry.shared.currentView else {
            result(FlutterError(code: "NO_VIEW", message: "AR view not ready.", details: nil))
            return
          }
          view.captureFrame(result: result)
        default:
          result(FlutterMethodNotImplemented)
        }
      }
    }
    return super.application(application, didFinishLaunchingWithOptions: launchOptions)
  }
}
