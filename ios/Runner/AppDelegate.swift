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
    let cameraPosition = SIMD3<Float>(
        cameraTransform.columns.3.x,
        cameraTransform.columns.3.y,
        cameraTransform.columns.3.z
    )
    
    var best: ClosestPlaneData?
    var bestDistance: Float = .infinity
    
    for anchor in frame.anchors {
        guard let plane = anchor as? ARPlaneAnchor, plane.alignment == .horizontal else { continue }
        
        // Get plane center in world coordinates
        let localCenter = SIMD4<Float>(plane.center.x, 0, plane.center.z, 1)
        let worldCenter4 = simd_mul(plane.transform, localCenter)
        let planePoint = SIMD3<Float>(worldCenter4.x, worldCenter4.y, worldCenter4.z)
        
        let planeNormal = simd_normalize(
            SIMD3<Float>(
                plane.transform.columns.1.x,
                plane.transform.columns.1.y,
                plane.transform.columns.1.z
            )
        )
        
        // For a horizontal plane, compute perpendicular distance from camera to plane
        // This is simply the height difference divided by the normal's Y component
        // which for horizontal planes is essentially the Y difference
        let heightDiff = cameraPosition.y - planePoint.y
        
        // Distance from camera to plane along viewing direction
        // For camera looking down at the table at angle θ from vertical:
        // distance = heightDiff / cos(θ)
        // We approximate this using the camera's actual viewing angle
        
        // Get the viewing direction's angle with vertical
        let cameraForward = -SIMD3<Float>(
            cameraTransform.columns.2.x,
            cameraTransform.columns.2.y,
            cameraTransform.columns.2.z
        )
        
        // Project camera forward onto plane normal to get cos(angle)
        let cosAngle = abs(simd_dot(cameraForward, planeNormal))
        
        guard cosAngle > 0.1 else { continue }  // Skip if nearly parallel
        
        // Distance along viewing direction to hit the plane
        let distance = abs(heightDiff) / cosAngle
        
        if distance < bestDistance && distance > 0.01 && distance < 2.0 {
            bestDistance = distance
            best = ClosestPlaneData(
                distanceMeters: Double(distance),
                center: planePoint,
                normal: planeNormal
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
