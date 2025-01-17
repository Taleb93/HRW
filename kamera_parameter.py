import pyrealsense2 as rs

def main():
    # Realsense-Pipeline initialisieren
    pipeline = rs.pipeline()
    config = rs.config()

    # Streams für RGB und Depth aktivieren
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # RGB-Stream
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)  # Depth-Stream

    # Pipeline starten
    profile = pipeline.start(config)
    print("Realsense-Kamera gestartet.")

    try:
        # Intrinsische Parameter des Depth-Sensors abrufen
        depth_sensor = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        depth_intrinsics = depth_sensor.get_intrinsics()

        print("\nIntrinsische Parameter des Depth-Sensors:")
        print(f"- Breite (Width): {depth_intrinsics.width}")
        print(f"- Höhe (Height): {depth_intrinsics.height}")
        print(f"- Brennweite X (fx): {depth_intrinsics.fx}")
        print(f"- Brennweite Y (fy): {depth_intrinsics.fy}")
        print(f"- Hauptpunkt X (ppx): {depth_intrinsics.ppx}")
        print(f"- Hauptpunkt Y (ppy): {depth_intrinsics.ppy}")
        print(f"- Verzerrungsmodell: {depth_intrinsics.model}")
        print(f"- Verzerrungskoeffizienten: {depth_intrinsics.coeffs}")

        # Prüfen, ob die Option unterstützt wird
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()

        if depth_sensor.supports(rs.option.max_distance):
            max_distance = depth_sensor.get_option(rs.option.max_distance)
            print(f"\nMaximale Reichweite der Depth-Kamera: {max_distance} Meter")
        else:
            print("\nDie maximale Reichweite ist für diesen Sensor nicht direkt abrufbar.")
            print("Standard-Reichweite: ca. 10 Meter (je nach Modell).")

    finally:
        pipeline.stop()
        print("\nRealsense-Kamera gestoppt.")

if __name__ == "__main__":
    main()
