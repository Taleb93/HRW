import argparse
import signal
import pyrealsense2 as rs
import time
import numpy as np
import cv2
from utils.utils import BaseEngine, preproc, vis
import Jetson.GPIO as GPIO

class Predictor(BaseEngine):
    def __init__(self, engine_path):
        super(Predictor, self).__init__(engine_path)
        self.n_classes = 9  # Anzahl der Klassen

# Globale Variablen für Signal-Handling
running = True

def signal_handler(signum, frame):
    """
    Signal-Handler, um das Programm sicher zu beenden.
    """
    global running
    print("\nSignal empfangen. Beende Prozess ...")
    running = False

# Signal-Handler für SIGINT (CTRL+C) und SIGTERM (Stop-Signal)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def GPIO_steuerung(GPIO_pin, final_cls_inds, final_scores, target_class_id, conf_threshold=0.5):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(GPIO_pin, GPIO.OUT)
    pin_high = False
    for cls_id, score in zip(final_cls_inds, final_scores):
        if int(cls_id) == target_class_id and score >= conf_threshold:
            pin_high = True
            break
    GPIO.output(GPIO_pin, GPIO.HIGH if pin_high else GPIO.LOW)

def GPIO_abstand_steuerung(GPIO_pin, depth_image, bounding_boxes, class_ids, scores, target_class_id, distance_threshold=1.0, default_conf_threshold=0.5):
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(GPIO_pin, GPIO.OUT)
    pin_high = False
    for box, cls_id, score in zip(bounding_boxes, class_ids, scores):
        if int(cls_id) == target_class_id:
            x1, y1, x2, y2 = map(int, box)
            depth_roi = depth_image[y1:y2, x1:x2]
            if score > default_conf_threshold:
                valid_depths = depth_roi[depth_roi > 0]
                avg_depth = np.mean(valid_depths) * 0.001 if valid_depths.size > 0 else float('inf')
                if avg_depth <= distance_threshold:
                    pin_high = True
                    break
    GPIO.output(GPIO_pin, GPIO.HIGH if pin_high else GPIO.LOW)

def vis_depth(depth_image, final_boxes, scores, conf=0.5):
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    for box, score in zip(final_boxes, scores):
        if score > conf:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(depth_colormap, (x1, y1), (x2, y2), (0, 255, 0), 2)
            depth_roi = depth_image[y1:y2, x1:x2]
            avg_depth = np.mean(depth_roi) * 0.001
            cv2.putText(depth_colormap, f"{avg_depth:.2f}m", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return depth_colormap

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--engine", help="TRT engine Path", default="best.trt")
    parser.add_argument("-i", "--image", help="image path", default=None)
    parser.add_argument("-o", "--output", help="image output path", default=None)
    parser.add_argument("-v", "--video", help="video path or camera index (use '0' for Realsense)", default='0')
    parser.add_argument("--end2end", default=True, action="store_true", help="use end2end engine")
    args = parser.parse_args()

    pred = Predictor(engine_path=args.engine)
    print(f"Loaded TensorRT engine from {args.engine}")

    if args.video:
        if args.video == '0':
            PIN = 7
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            pipeline.start(config)
            align = rs.align(rs.stream.color)
            print("Using Realsense camera...")

            try:
                while running:
                    frames = pipeline.wait_for_frames()
                    aligned_frames = align.process(frames)
                    aligned_depth_frame = aligned_frames.get_depth_frame()
                    color_frame = aligned_frames.get_color_frame()

                    if not color_frame or not aligned_depth_frame:
                        print("No frames received from Realsense!")
                        continue

                    color_image = np.asanyarray(color_frame.get_data())
                    depth_image = np.asanyarray(aligned_depth_frame.get_data())

                    print(f"RGB Image Size: {color_image.shape}, Depth Image Size: {depth_image.shape}")

                    blob, ratio = preproc(color_image, pred.imgsz, pred.mean, pred.std)
                    t1 = time.time()
                    data = pred.infer(blob)
                    fps = 1.0 / (time.time() - t1)
                    print(f"Inference FPS: {fps:.2f}")

                    if args.end2end:
                        num, final_boxes, final_scores, final_cls_inds = data
                        final_boxes = np.reshape(final_boxes / ratio, (-1, 4))
                    else:
                        predictions = np.reshape(data, (1, -1, int(5 + pred.n_classes)))[0]
                        dets = pred.postprocess(predictions, ratio)
                        final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]

                    color_image = vis(color_image, final_boxes, final_scores, final_cls_inds, conf=0.5, class_names=pred.class_names)
                    depth_colormap = vis_depth(depth_image, final_boxes, final_scores, conf=0.5)

                    GPIO_abstand_steuerung(32, depth_image, final_boxes, final_cls_inds, final_scores, 3, 1, 0.1)
                    GPIO_abstand_steuerung(33, depth_image, final_boxes, final_cls_inds, final_scores, 6, 1, 0.1)
                    GPIO_abstand_steuerung(7, depth_image, final_boxes, final_cls_inds, final_scores, 7, 1, 0.1)

                    #cv2.imshow("RGB Frame", color_image)
                    #cv2.imshow("Depth Frame", depth_colormap)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            except Exception as e:
                print(f"Ein Fehler ist aufgetreten: {e}")
            finally:
                print("Aufräumen und Beenden...")
                GPIO.cleanup()
                try:
                    pipeline.stop()
                except RuntimeError as e:
                    print(f"Fehler beim Stoppen der Pipeline: {e}")
                cv2.destroyAllWindows()
        else:
            pred.detect_video(args.video, conf=0.5, end2end=args.end2end)
    else:
        print("Error: Specify either an image (-i), video (-v), or Realsense camera (use -v 0).")
