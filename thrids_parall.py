import threading
import queue
from queue import Empty
import pycuda.driver as cuda
import pycuda.autoinit  # Nur für den Hauptthread
import numpy as np
import cv2
import os
from utils.utils import BaseEngine, preproc, vis, AutoencoderEngine, cuda_init,optimalYolo
import torchvision.transforms as transforms
from PIL import Image
import time

# Globale Queues für Thread-Kommunikation
yolo_to_autoencoder_queue = queue.Queue(maxsize=10)
autoencoder_to_yolo_queue = queue.Queue(maxsize=10)

def preprocess_image_array(image_array):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
    return transform(image).unsqueeze(0).numpy()

def denormalize(image_tensor):
    return (image_tensor * 0.5 + 0.5).clip(0, 1)


def ensure_output_dir(path):
    """Stellt sicher, dass ein Ausgabeordner existiert."""
    if not os.path.exists(path):
        os.makedirs(path)

def put_object_in_queue(final_boxes, final_scores, final_cls_inds, color_image, queue, valid_classes, score_range=(0.5, 1.0)):
    """
    Filtert Objekte basierend auf Wahrscheinlichkeiten und Klassen, erstellt quadratische Bounding-Boxes 
    und fügt zugeschnittene Bilder in die Queue ein.
    """
    for box, score, cls_id in zip(final_boxes, final_scores, final_cls_inds):
        if cls_id not in valid_classes:
            print("cropped image nicht valid_classes.")
            continue
        if not (score_range[0] <= score <= score_range[1]):
            print("cropped image nicht erstellt.")
            continue

        x1, y1, x2, y2 = map(int, box)

        # Quadratbildung der Bounding-Box
        width, height = x2 - x1, y2 - y1
        size = max(width, height)
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        new_x1 = max(0, center_x - size // 2)
        new_y1 = max(0, center_y - size // 2)
        new_x2 = min(color_image.shape[1], center_x + size // 2)
        new_y2 = min(color_image.shape[0], center_y + size // 2)

        cropped_image = color_image[new_y1:new_y2, new_x1:new_x2]
        print("cropped image erstellt.")

        if cropped_image.size == 0:
            continue

        try:
            queue.put((cropped_image, box), timeout=1)
        except queue.Full:
            print("Queue ist voll. Überspringe das Bild.")

def yolo_inference_on_image_thread(image_path, engine_path, output_dir, queue_out, valid_classes, score_range=(0.5, 1.0)):
    """YOLO-Inferenz und Objektübertragung an die Autoencoder-Queue."""
    context1 = None
    context1 = cuda_init()  # Initialisierung des CUDA-Kontexts für diesen Thread
    try:
        yolo_predictor = BaseEngine(engine_path)
        print(f"YOLO: TensorRT-Engine {engine_path} geladen.")

        # Bild laden
        color_image = cv2.imread(image_path)
        if color_image is None:
            print(f"Bild {image_path} konnte nicht geladen werden.")
            return

        blob, ratio = preproc(color_image, yolo_predictor.imgsz, yolo_predictor.mean, yolo_predictor.std)

        # YOLO-Inferenz
        t1 = time.time()
        data = yolo_predictor.infer(blob)
        fps = 1.0 / (time.time() - t1)
        print(f"YOLO: Inferenz abgeschlossen. FPS: {fps:.2f}")

        # Ergebnisse verarbeiten
        num, final_boxes, final_scores, final_cls_inds = data
        final_boxes = np.reshape(final_boxes / ratio, (-1, 4))
        dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]

        # **Hier wird put_object_in_queue aufgerufen**
        put_object_in_queue(
            final_boxes=final_boxes,
            final_scores=final_scores,
            final_cls_inds=final_cls_inds,
            color_image=color_image,
            queue=queue_out,
            valid_classes=valid_classes,
            score_range=score_range,
        )

        # YOLO-Ergebnisse visualisieren und speichern
        vis_image = vis(color_image, final_boxes, final_scores, final_cls_inds, conf=0.5, class_names=yolo_predictor.class_names)
        vis_output_path = os.path.join(output_dir, "yolo_visualization.png")
        cv2.imwrite(vis_output_path, vis_image)
        print(f"Visualisiertes YOLO-Bild gespeichert unter: {vis_output_path}")

    finally:
         queue_out.put(None)
          # Stopp-Signal senden
         print("yolo1: Stopp-Signal gesendet. Beende nächste Thread.")
         if yolo_predictor:
            print("YOLO: Freigabe der YOLO-Engine.")
            del yolo_predictor
         if context1:
            context1.pop()
            del context1
         print("YOLO: Kontext freigegeben.")


def autoencoder_thread(engine_path, input_queue, output_queue, output_dir):
    """Autoencoder verarbeitet Objekte aus der Queue und sendet rekonstruierte Bilder."""
    context2 = None
    context2 = cuda_init()  # Initialisierung des CUDA-Kontexts für diesen Thread
    try:
        autoencoder = AutoencoderEngine(engine_path)
        #print(f"Autoencoder: TensorRT-Engine {engine_path} geladen.")

        while True:
            try:
                data = input_queue.get(timeout=1)
                if data is None:
                    print("Autoencoder: Stopp-Signal empfangen von thread1. Beende Thread2.")
                      # Weiteres Stopp-Signal senden
                    break

                cropped_image, box = data
                preprocessed_image = preprocess_image_array(cropped_image)
                output, _ = autoencoder.infer(preprocessed_image)

                # Rekonstruiertes Bild speichern und weitergeben
                out_image = denormalize(output[0].transpose(1, 2, 0))
                out_image = np.clip(out_image, 0, 1)*255
                print("autoencoder image wurde im queue gesetzt")
                output_queue.put((out_image.astype(np.uint8), box))

                display_image = (out_image).astype(np.uint8)

                # Rekonstruiertes Bild speichern
                recon_output_path = os.path.join(output_dir, f"reconstructed_{time.time():.0f}.png")
                cv2.imwrite(recon_output_path, cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR))
                print(f"Rekonstruiertes Bild gespeichert unter: {recon_output_path}")


            except Empty:
                continue
            except Exception as e:
                print(f"Autoencoder: Fehler: {e}")

    finally:
        #output_queue.put(None)
        if autoencoder:
            print("autoencoder: Freigabe der autoencoder-Engine.")
            del autoencoder
        if context2:
            time.sleep(1)
            output_queue.put(None)
            print("Autoencoder: Stopp-Signal gesendet von thread2. Beende Thread3.")
            context2.pop()
            del context2
        print("Autoencoder: Kontext freigegeben.")


def creat_image_for_yolo_thread(output_queue, image_path, engine_yolo_path, yolo_results_dir):
    """Fügt rekonstruierte Bilder in das Originalbild ein und führt erneut YOLO aus."""
    context3 = None
    context3 = cuda_init()  # Initialisierung des CUDA-Kontexts für diesen Thread
    try:
        yolo_predictor = BaseEngine(engine_yolo_path)
        print(f"YOLO: TensorRT-Engine für Rekonstruktion {engine_yolo_path} geladen.")

        original_image = cv2.imread(image_path)
        if original_image is None:
            print(f"Bild {image_path} konnte nicht geladen werden.")
            return

        while True:
            try:
                data = output_queue.get(timeout=50)
                print("autoencoder image wurde von queue geholt für das 3 thrids")
                if data is None:
                    #if context:
                    print("YOLO-Rekonstruktion: Stopp-Signal empfangen. Beende Thread3.")
                    break

                display_image, box = data
                x1, y1, x2, y2 = map(int, box)
                box_width = x2 - x1
                box_height = y2 - y1

                # Größe des rekonstruierten Bildes anpassen
                if box_width > 64 or box_height > 64:
                    # Falls Bounding-Box größer als 64x64 ist, skalieren wir das rekonstruierte Bild hoch
                    scale_width = max(box_width, 64)
                    scale_height = max(box_height, 64)
                    display_image = cv2.resize(display_image, (scale_width, scale_height), interpolation=cv2.INTER_LINEAR)
                    #resized_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2BGR)
                
                #if len(resized_image.shape) == 2:  # Falls das Bild plötzlich schwarz-weiß ist
                
                # Setze das skalierte Bild in das Originalbild ein
                h, w, _ = display_image.shape
                start_y = max(0, y1 + (box_height - h) // 2)
                start_x = max(0, x1 + (box_width - w) // 2)
                end_y = min(original_image.shape[0], start_y + h)
                end_x = min(original_image.shape[1], start_x + w)

                # Sicherstellen, dass die Dimensionen übereinstimmen
                cropped_region = original_image[start_y:end_y, start_x:end_x]
                display_image = cv2.resize(display_image, (cropped_region.shape[1], cropped_region.shape[0]))
                original_image[start_y:end_y, start_x:end_x] = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)

                # YOLO-Inferenz auf dem modifizierten Bild
                blob, ratio = preproc(original_image, yolo_predictor.imgsz, yolo_predictor.mean, yolo_predictor.std)
                t1 = time.time()
                data = yolo_predictor.infer(blob)
                fps = 1.0 / (time.time() - t1)
                print(f"YOLO: Inferenz abgeschlossen. FPS: {fps:.2f}")

                # Ergebnisse verarbeiten und speichern
                num, final_boxes, final_scores, final_cls_inds = data
                final_boxes = np.reshape(final_boxes / ratio, (-1, 4))

                vis_image = vis(original_image, final_boxes, final_scores, final_cls_inds, conf=0.5, class_names=yolo_predictor.class_names)
                yolo_output_path = os.path.join(yolo_results_dir, f"yolo_reconstructed_{time.time():.0f}.png")
                cv2.imwrite(yolo_output_path, vis_image)
                print(f"YOLO-Ergebnis mit Rekonstruktion gespeichert unter: {yolo_output_path}")


                # # Bildgröße anpassen und einfügen
                # resized_image = cv2.resize(display_image, (x2 - x1, y2 - y1), interpolation=cv2.INTER_LINEAR)
                # original_image[y1:y2, x1:x2] = resized_image

                # # YOLO-Inferenz auf dem modifizierten Bild
                # blob, ratio = preproc(original_image, yolo_predictor.imgsz, yolo_predictor.mean, yolo_predictor.std)
                # data = yolo_predictor.infer(blob)

                # num, final_boxes, final_scores, final_cls_inds = data
                # final_boxes = np.reshape(final_boxes / ratio, (-1, 4))

                # vis_image = vis(original_image, final_boxes, final_scores, final_cls_inds, conf=0.5, class_names=yolo_predictor.class_names)
                # yolo_output_path = os.path.join(yolo_results_dir, f"yolo_reconstructed_{time.time():.0f}.png")
                # cv2.imwrite(yolo_output_path, vis_image)


            except Empty:
                print("nicht geschfft")
                continue
            except Exception as e:
                print(f"YOLO-Rekonstruktion: Fehler: {e}")

    finally:
        if yolo_predictor:
            print("YOLO-Rekonstruktion: Freigabe der YOLO-Engine.")
            del yolo_predictor
        if context3:
            context3.pop()
            del context3
        else:
            print("context berets beendet")
        print("YOLO-Rekonstruktion: Kontext freigegeben.")


def main():
    engine_yolo_path = "best.trt"
    #engine_yolo_path1 = "best1.trt"
    engine_autoencoder_path = "best_masked.trt"
    image_path = "T62.jpg"
    output_dir = "output_images"
    yolo_results_dir = "yolo_results"

    ensure_output_dir(output_dir)
    ensure_output_dir(yolo_results_dir)

    # Threads erstellen
    yolo_thread = threading.Thread(
        target=yolo_inference_on_image_thread,
        args=(image_path, engine_yolo_path, output_dir, yolo_to_autoencoder_queue, [3, 4, 5, 6, 7, 8])
    )
    autoencoder_thread_instance = threading.Thread(
        target=autoencoder_thread,
        args=(engine_autoencoder_path, yolo_to_autoencoder_queue, autoencoder_to_yolo_queue, output_dir)
    )
    
    reconstruction_thread = threading.Thread(
        target=creat_image_for_yolo_thread,
        args=(autoencoder_to_yolo_queue, image_path, engine_yolo_path, yolo_results_dir)
    )

    # Threads starten
        # Threads starten mit Pausen
    yolo_thread.start()
    print("YOLO-Thread gestartet.")
    time.sleep(1)  # 1 Sekunde Pause, um sicherzustellen, dass der YOLO-Thread vollständig läuft

    autoencoder_thread_instance.start()
    print("Autoencoder-Thread gestartet.")
    time.sleep(1)  # 1 Sekunde Pause, um sicherzustellen, dass der Autoencoder-Thread läuft

    reconstruction_thread.start()
    print("Rekonstruktions-Thread gestartet.")

    # Warten auf Threads
    yolo_thread.join()
    autoencoder_thread_instance.join()
    reconstruction_thread.join()
    print("Programm beendet.")


if __name__ == "__main__":
    main()