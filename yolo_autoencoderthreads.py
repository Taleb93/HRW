import threading
import time
import queue
from queue import Queue, Empty
#import pycuda.driver as cuda
import torchvision.transforms as transforms
#import pycuda.autoinit  # Nur für den Hauptthread
import numpy as np
import pyrealsense2 as rs
import cv2
from utils.utils import BaseEngine, preproc, vis, AutoencoderEngine,cuda_init
from PIL import Image

# Globale Queues für Thread-Kommunikation
yolo_to_autoencoder_queue = queue.Queue(maxsize=10)
autoencoder_to_yolo_queue = queue.Queue(maxsize=10)
def put_object_in_queue(final_boxes, final_scores, final_cls_inds, color_image, queue, valid_classes, score_range=(0.5, 0.7)):
    """
    Filtert Objekte basierend auf Wahrscheinlichkeiten und Klassen, erstellt quadratische Bounding-Boxes 
    und fügt zugeschnittene Bilder in die Queue ein.
    
    Args:
        final_boxes (np.ndarray): Bounding-Box-Koordinaten (x1, y1, x2, y2).
        final_scores (np.ndarray): Wahrscheinlichkeiten der Objekte.
        final_cls_inds (np.ndarray): Klassenindizes der Objekte.
        color_image (np.ndarray): Eingabebild.
        queue (queue.Queue): Queue für die Objekte.
        valid_classes (list): Liste der Klassen, die in die Queue eingefügt werden sollen.
        score_range (tuple): Min- und Max-Wahrscheinlichkeitsbereich (inklusive).
    """
    for box, score, cls_id in zip(final_boxes, final_scores, final_cls_inds):
        # Überprüfe, ob die Klasse gültig ist
        if cls_id not in valid_classes:
            continue
        
        # Überprüfe, ob die Wahrscheinlichkeit im gewünschten Bereich liegt
        if not (score_range[0] <= score <= score_range[1]):
            continue

        x1, y1, x2, y2 = map(int, box)

        # Validierung der Bounding-Box
        if x2 <= x1 or y2 <= y1:
            print(f"Ungültige Bounding-Box übersprungen: {box}")
            continue

        # Quadratbildung der Bounding-Box
        width = x2 - x1
        height = y2 - y1
        size = max(width, height)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

        # Berechne die neuen quadratischen Koordinaten
        new_x1 = max(0, center_x - size // 2)
        new_y1 = max(0, center_y - size // 2)
        new_x2 = min(color_image.shape[1], center_x + size // 2)
        new_y2 = min(color_image.shape[0], center_y + size // 2)

        # Validierung der neuen Bounding-Box
        if new_x2 <= new_x1 or new_y2 <= new_y1:
            print(f"Quadratische Bounding-Box ist ungültig: {new_x1, new_y1, new_x2, new_y2}")
            continue

        # Zuschneiden des Bildes
        cropped_image = color_image[new_y1:new_y2, new_x1:new_x2]

        # Validierung des zugeschnittenen Bildes
        if cropped_image is None or cropped_image.size == 0:
            print("Leeres Bild entdeckt. Überspringe Verarbeitung.")
            continue

        # Einfügen in die Queue
        try:
            queue.put((cropped_image, box), timeout=1)
        except queue.Full:
            print("Queue ist voll. Überspringe das Bild.")
            continue

def preprocess_image_array(image_array):
    """
    Bereitet ein Numpy-Array-Bild für den Autoencoder vor:
    - Konvertierung in RGB (falls nötig).
    - Größenänderung auf 64x64.
    - Normalisierung der Pixelwerte auf [-1, 1].
    - Formatumwandlung in (C, H, W) mit Batch-Dimension.
    """
    # Konvertiere von BGR (OpenCV) zu RGB, falls nötig
    if image_array.shape[2] == 3:  # Sicherstellen, dass es ein Farbbild ist
        image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    else:
        image = Image.fromarray(image_array)

    # Bildgröße ändern und normalisieren
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),  # Konvertiert in Tensor (C, H, W)
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalisierung auf [-1, 1]
    ])

    input_tensor = transform(image).unsqueeze(0).numpy()  # Batch-Dimension hinzufügen
    return input_tensor


def denormalize(image_tensor):
    return (image_tensor * 0.5 + 0.5).clip(0, 1)

def insert_object_into_scene(original_image, reconstructed_image, bounding_box):
    """
    Fügt das rekonstruierte Objekt in die Originalszene ein.

    Args:
        original_image (np.ndarray): Das Originalbild (RGB, 640x640).
        reconstructed_image (np.ndarray): Das rekonstruierte Objekt (RGB, z. B. 64x64).
        bounding_box (tuple): Koordinaten der Bounding-Box (x1, y1, x2, y2).

    Returns:
        np.ndarray: Das Bild mit dem eingefügten Objekt.
    """
    x1, y1, x2, y2 = map(int, bounding_box)

    # Berechne die Größe der Bounding-Box
    box_width = max(1, x2 - x1)
    box_height = max(1, y2 - y1)

    # Überprüfe, ob Bounding-Box gültig ist
    if box_width <= 0 or box_height <= 0:
        print("Ungültige Bounding-Box, überspringe Einfügen.")
        return original_image

    # Skaliere das rekonstruierte Objekt auf die Größe der Bounding-Box
    resized_object = cv2.resize(reconstructed_image, (box_width, box_height), interpolation=cv2.INTER_LINEAR)

    # Begrenze die Koordinaten der Bounding-Box auf die Bildgrenzen
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(original_image.shape[1], x2), min(original_image.shape[0], y2)

    # Prüfe, ob die Größen immer noch übereinstimmen
    if resized_object.shape[0] != (y2 - y1) or resized_object.shape[1] != (x2 - x1):
        print(f"Warnung: Größenkonflikt nach Resize! resized_object: {resized_object.shape}, Bounding-Box: {(y2-y1, x2-x1)}")
        resized_object = resized_object[:y2 - y1, :x2 - x1]

    # Ersetze den Bereich im Originalbild
    updated_image = original_image.copy()
    updated_image[y1:y2, x1:x2] = cv2.cvtColor(resized_object, cv2.COLOR_RGB2BGR)

    return updated_image



# YOLO-Inferenz-Thread
def yolo_inference_thread(engine_path,static_background, running_event):
    # Manuelle CUDA-Kontextinitialisierung
    context = cuda_init()

    try:
        # TensorRT-Modell laden
        yolo_predictor = BaseEngine(engine_path)
        print(f"YOLO-Thread: TensorRT-Engine {engine_path} geladen.")


        # RealSense-Kamera initialisieren
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        align = rs.align(rs.stream.color)
        print("RealSense-Kamera gestartet.")

        while running_event.is_set():
            try:

                if not autoencoder_to_yolo_queue.empty():
                    #try:
                        # Hole rekonstruiertes Bild aus der Queue
                        reconstructed_image, bounding_box = autoencoder_to_yolo_queue.get(timeout=0.1)
                        updated_scene = insert_object_into_scene(
                            original_image=static_background,
                            reconstructed_image=reconstructed_image,
                            bounding_box=bounding_box
                        )

                        # Validierung des aktualisierten Bildes
                        #if updated_scene.dtype != np.uint8 or len(updated_scene.shape) != 3 or updated_scene.shape[2] != 3:
                            #raise ValueError("updated_scene ist kein gültiges RGB-Bild.")

                        # YOLO-Inferenz auf rekonstruiertem Bild
                        blob, ratio = preproc(updated_scene, yolo_predictor.imgsz, yolo_predictor.mean, yolo_predictor.std)
                        data = yolo_predictor.infer(blob)

                        # Ergebnisse verarbeiten
                        num, final_boxes, final_scores, final_cls_inds = data
                        final_boxes = np.reshape(final_boxes / ratio, (-1, 4))

                        # Visualisieren
                        vis_image = vis(updated_scene, final_boxes, final_scores, final_cls_inds, conf=0.5, class_names=yolo_predictor.class_names)
                        cv2.imshow("YOLO on Reconstructed Image", vis_image)

                    #except Exception as e:
                        #print(f"Fehler bei der Verarbeitung des rekonstruierten Objekts: {e}")

                # Kameraframes abrufen
                frames = pipeline.wait_for_frames()
                aligned_frames = align.process(frames)
                color_frame = aligned_frames.get_color_frame()

                if not color_frame:
                    print("Kein Farbframe verfügbar!")
                    continue

                # Bilddaten abrufen und YOLO-Inferenz ausführen
                color_image = np.asanyarray(color_frame.get_data())
                blob, ratio = preproc(color_image, yolo_predictor.imgsz, yolo_predictor.mean, yolo_predictor.std)

                # YOLO-Inferenz
                t1 = time.time()
                data = yolo_predictor.infer(blob)
                fps = 1.0 / (time.time() - t1)
                print(f"YOLO-Thread: Inferenz abgeschlossen. FPS: {fps:.2f}")

                # Ergebnisse verarbeiten
                num, final_boxes, final_scores, final_cls_inds = data
                final_boxes = np.reshape(final_boxes / ratio, (-1, 4))

                # Definiere gültige Klassen (nur 1 bis 5)
                valid_classes = [3, 4, 5, 6, 7, 8]

                # Verwende die Funktion
                put_object_in_queue(
                    final_boxes=final_boxes,
                    final_scores=final_scores,
                    final_cls_inds=final_cls_inds,
                    color_image=color_image,
                    queue=yolo_to_autoencoder_queue,
                    valid_classes=valid_classes
                )


                vis_image = vis(color_image, final_boxes, final_scores, final_cls_inds, conf=0.5, class_names=yolo_predictor.class_names)

                # Bild anzeigen
                cv2.imshow("YOLO Inference", vis_image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running_event.clear()
                    break

            except Empty:
                continue

            except Exception as e:
                print(f"YOLO-Thread: Fehler während der Inferenz: {e}")
                break

        pipeline.stop()
        print("RealSense-Kamera gestoppt.")
    finally:
        # Kontext freigeben
       
           if context:
              context.pop()
              del context
           print("YOLO-Thread: Kontext freigegeben.")

        

# # Autoencoder-Inferenz-Thread
def autoencoder_thread(engine_path, input_queue, running_event):
    # Manuelle CUDA-Kontextinitialisierung
    context = cuda_init()

    try:
        # TensorRT-Engine laden
        autoencoder = AutoencoderEngine(engine_path)
        print(f"Autoencoder-Thread: TensorRT-Engine {engine_path} geladen.")

        while running_event.is_set():
            try:
                # Eingabe aus der Queue abrufen
                cropped_image, box = input_queue.get(timeout=1)
                print(f"Bildform vor preprocess_image_array: {cropped_image.shape}, Min={cropped_image.min()}, Max={cropped_image.max()}")
                

                preprocessed_image = preprocess_image_array(cropped_image)
                print(f"Bildform nach preprocess_image_array: {preprocessed_image.shape}, Min={preprocessed_image.min()}, Max={preprocessed_image.max()}")

                # Inferenz ausführen
                output, inference_time = autoencoder.infer(preprocessed_image)
                print(f"Bildform nach autoencoder.infer: {output.shape}, Min={output.min()}, Max={output.max()}")

                out_image = denormalize(output[0].transpose(1, 2, 0))  # (C, H, W) -> (H, W, C)
                print(f"Bildform nach denormalize: {out_image.shape}, Min={out_image.min()}, Max={out_image.max()}")
                out_image = np.clip(out_image, 0, 1)  # Sicherheitshalber einschränken
                print(f"Bildform nach np.clip: {out_image.shape}, Min={out_image.min()}, Max={out_image.max()}")
                if np.isnan(out_image).any():
                    print("NaN-Werte in der Autoencoder-Ausgabe gefunden!")
                    continue
                if out_image.min() < 0 or out_image.max() > 1:
                    print(f"Unerwartete Werte in der Autoencoder-Ausgabe: min={out_image.min()}, max={out_image.max()}")

                # Konvertieren für OpenCV
                #display_image = (out_image * 255).clip(0, 255).astype(np.uint8)

                display_image = (out_image * 255).astype(np.uint8)

                try:
                    autoencoder_to_yolo_queue.put((display_image, box), timeout=1)
                    print("Rekonstruiertes Bild in autoencoder_to_yolo_queue eingefügt.")
                except queue.Full:
                    print("autoencoder_to_yolo_queue ist voll. Rekonstruktion wird verworfen.")

                #display_image = cv2.normalize(display_image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                #bgr_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
                # Debugging-Ausgaben
                #print(f"Rekonstruierte Bildform: {bgr_image.shape}, Min={bgr_image.min()}, Max={bgr_image.max()}")
                #print(f"Rekonstruierte Bildform vor Anzeige: {display_image.shape}, Min={display_image.min()}, Max={display_image.max()}")
                #cv2.imshow("Reconstructed Image", bgr_image)
                
                print(f"Autoencoder-Thread: Rekonstruktion abgeschlossen. Inferenzzeit: {inference_time:.2f} ms")
            except Empty:
                continue  # Keine Daten in der Queue, weitermachen
            except Exception as e:
                print(f"Autoencoder-Thread: Fehler während der Inferenz: {e}")
                continue
    finally:
        # Kontext freigeben
        if autoencoder:
            print("autoencoder: Freigabe der autoencoder-Engine.")
            del autoencoder
    
        if context:
            context.pop()
            del context
            print("AutoEncoder-Thread: Kontext freigegeben.")
           
        
def main():
    engine_yolo_path = "best.trt"  # Pfad zur YOLO-Engine
    engine_autoencoder_path = "best_masked.trt"  # Pfad zur Autoencoder-Engine
    background_path = "static_background.jpg"
    
    # Hintergrundbild einmalig laden und skalieren
    static_background = cv2.imread(background_path)  # Bild laden
    if static_background is None:
        raise ValueError(f"Konnte das Hintergrundbild nicht laden: {background_path}")
    static_background = cv2.resize(static_background, (640, 640))  # YOLO-kompatible Größe
    print("Hintergrundbild erfolgreich geladen und skaliert.")
    running_event = threading.Event()
    running_event.set()

    # YOLO- und Autoencoder-Threads starten
    yolo_thread = threading.Thread(target=yolo_inference_thread, args=(engine_yolo_path,static_background, running_event))
    autoencoder_thread2 = threading.Thread(target=autoencoder_thread, args=(engine_autoencoder_path, yolo_to_autoencoder_queue, running_event))
    yolo_thread.start()
    autoencoder_thread2.start()

    try:
        while True:
            time.sleep(1)  # Hauptthread aktiv halten
    except KeyboardInterrupt:
        print("Beenden signalisiert.")
        running_event.clear()
        yolo_thread.join()
        autoencoder_thread2.join()
        print("Programm beendet.")

if __name__ == "__main__":
    main()
