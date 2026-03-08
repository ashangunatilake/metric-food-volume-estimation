import 'dart:convert';
import 'dart:io';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:http/http.dart' as http;
import 'package:http_parser/http_parser.dart';

const String kDefaultEndpoint = 'http://192.168.1.4:8000/estimate_volume';

void main() {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const ResearchClientApp());
}

class ResearchClientApp extends StatelessWidget {
  const ResearchClientApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Research Client',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorSchemeSeed: const Color(0xFF0D3B66),
      ),
      home: const CaptureScreen(),
    );
  }
}

class CaptureScreen extends StatefulWidget {
  const CaptureScreen({super.key});

  @override
  State<CaptureScreen> createState() => _CaptureScreenState();
}

class _CaptureScreenState extends State<CaptureScreen> {
  static const MethodChannel _channel = MethodChannel('arkit_capture');

  final TextEditingController _endpointController =
      TextEditingController(text: kDefaultEndpoint);

  bool _isBusy = false;
  String _status = 'Ready.';
  double? _volumeCm3;
  double? _planeDistanceM;
  Uint8List? _capturedImage;

  @override
  void dispose() {
    _endpointController.dispose();
    super.dispose();
  }

  Future<void> _captureAndSend() async {
    if (!Platform.isIOS) {
      setState(() {
        _status = 'ARKit is only available on iOS.';
      });
      return;
    }

    final endpoint = _endpointController.text.trim();
    if (endpoint.isEmpty || endpoint.contains('<your-backend-url>')) {
      setState(() {
        _status = 'Set your backend endpoint before sending.';
      });
      return;
    }

    setState(() {
      _isBusy = true;
      _status = 'Capturing frame...';
      _volumeCm3 = null;
      _planeDistanceM = null;
      _capturedImage = null;
    });

    try {
      final Map<dynamic, dynamic> payload =
          await _channel.invokeMethod('captureFrame');

      final double fx = (payload['fx'] as num).toDouble();
      final double fy = (payload['fy'] as num).toDouble();
      final double cx = (payload['cx'] as num).toDouble();
      final double cy = (payload['cy'] as num).toDouble();
      final double planeDistance =
          (payload['plane_distance_m'] as num).toDouble();
      final List<dynamic> planeCenterRaw = payload['plane_center'] as List<dynamic>;
      final List<dynamic> planeNormalRaw = payload['plane_normal'] as List<dynamic>;
      final List<double> planeCenter =
          planeCenterRaw.map((v) => (v as num).toDouble()).toList(growable: false);
      final List<double> planeNormal =
          planeNormalRaw.map((v) => (v as num).toDouble()).toList(growable: false);
      final Uint8List imageBytes = payload['image_bytes'] as Uint8List;

      setState(() {
        _status = 'Uploading to backend...';
      });

      final uri = Uri.parse(endpoint);
      final request = http.MultipartRequest('POST', uri);
      request.fields['plane_distance_m'] = planeDistance.toString();
      request.fields['fx'] = fx.toString();
      request.fields['fy'] = fy.toString();
      request.fields['cx'] = cx.toString();
      request.fields['cy'] = cy.toString();
      request.fields['plane_center'] = jsonEncode(planeCenter);
      request.fields['plane_normal'] = jsonEncode(planeNormal);
      request.files.add(
        http.MultipartFile.fromBytes(
          'image',
          imageBytes,
          filename: 'frame.jpg',
          contentType: MediaType('image', 'jpeg'),
        ),
      );

      final response = await request.send();
      final responseBody = await response.stream.bytesToString();

      if (response.statusCode != 200) {
        throw Exception('Server error (${response.statusCode}): $responseBody');
      }

      final decoded = jsonDecode(responseBody) as Map<String, dynamic>;
      final volume = (decoded['volume_cm3'] as num).toDouble();

      setState(() {
        _volumeCm3 = volume;
        _planeDistanceM = planeDistance;
        _capturedImage = imageBytes;
        _status = 'Success.';
      });
    } on PlatformException catch (e) {
      setState(() {
        _status = 'Native error: ${e.code} ${e.message ?? ''}'.trim();
      });
    } catch (e) {
      setState(() {
        _status = 'Error: $e';
      });
    } finally {
      setState(() {
        _isBusy = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('AR Volume Capture'),
      ),
      body: Column(
        children: [
          Expanded(
            child: Container(
              color: Colors.black,
              child: Platform.isIOS
                  ? const UiKitView(viewType: 'arkit_view')
                  : const Center(
                      child: Text(
                        'ARKit view is only available on iOS.',
                        style: TextStyle(color: Colors.white),
                      ),
                    ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 12, 16, 8),
            child: TextField(
              controller: _endpointController,
              keyboardType: TextInputType.url,
              decoration: const InputDecoration(
                labelText: 'Endpoint',
                hintText: 'https://your-app.azurecontainerapps.io/estimate_volume',
                border: OutlineInputBorder(),
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
            child: Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: _isBusy ? null : _captureAndSend,
                    child: Text(_isBusy ? 'Working...' : 'Capture + Send'),
                  ),
                ),
                const SizedBox(width: 12),
                if (_volumeCm3 != null)
                  Text(
                    '${_volumeCm3!.toStringAsFixed(1)} cm³',
                    style: const TextStyle(
                      fontSize: 18,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
              ],
            ),
          ),
          if (_capturedImage != null && _planeDistanceM != null && _volumeCm3 != null)
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 0, 16, 12),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  ClipRRect(
                    borderRadius: BorderRadius.circular(8),
                    child: Image.memory(
                      _capturedImage!,
                      width: 96,
                      height: 96,
                      fit: BoxFit.cover,
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Plane distance: ${_planeDistanceM!.toStringAsFixed(3)} m',
                          style: const TextStyle(fontSize: 14),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          'Volume: ${_volumeCm3!.toStringAsFixed(1)} cm³',
                          style: const TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
            child: Align(
              alignment: Alignment.centerLeft,
              child: Text(
                _status,
                style: TextStyle(
                  color: _status.startsWith('Error') || _status.startsWith('Native')
                      ? Colors.red
                      : Colors.black87,
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
