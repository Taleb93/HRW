import tensorrt as trt
import pycuda.autoinit
import pycuda.driver as cuda
import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
class BaseEngine(object):
    def __init__(self, engine_path):
        self.mean = None
        self.std = None
        self.n_classes = 9
        self.class_names = [
            "Ampel-gelb", "Ampel-grun", "Ampel-rot", "Geradeaus",
            "Geradeaus-oder-links", "Geradeaus-oder-rechts",
            "Links-abbiegen", "Rechts-abbiegen", "Zebrastreifen"
        ]

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        trt.init_libnvinfer_plugins(logger, '')  # TensorRT-Plugins initialisieren
        with open(engine_path, "rb") as f:
            serialized_engine = f.read()
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.imgsz = engine.get_binding_shape(0)[2:]  # Eingabegröße des Modells
        self.context = engine.create_execution_context()
        self.inputs, self.outputs, self.bindings = [], [], []
        self.stream = cuda.Stream()

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append(int(device_mem))
            if engine.binding_is_input(binding):
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, img):
        self.inputs[0]['host'] = np.ravel(img)
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        data = [out['host'] for out in self.outputs]
        return data

    def get_fps(self):
        """FPS messen, indem Inferenz mehrere Male durchgeführt wird."""
        import time
        img = np.ones((1, 3, self.imgsz[0], self.imgsz[1]), dtype=np.float32)
        for _ in range(5):  # Warm-up-Runden
            _ = self.infer(img)

        t0 = time.perf_counter()
        for _ in range(100):  # Zeit für 100 Inferenzen messen
            _ = self.infer(img)
        fps = 100 / (time.perf_counter() - t0)
        print(f"{fps:.2f} FPS")
        return fps

    def detect_video(self, video_path, conf=0.5, end2end=False):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = int(round(cap.get(cv2.CAP_PROP_FPS)))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter('results.avi', fourcc, fps, (width, height))
        fps = 0
        import time
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            blob, ratio = preproc(frame, self.imgsz, self.mean, self.std)
            t1 = time.time()
            data = self.infer(blob)
            fps = (fps + (1. / (time.time() - t1))) / 2
            frame = cv2.putText(frame, f"FPS: {fps:.2f}", (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            if end2end:
                num, final_boxes, final_scores, final_cls_inds = data
                final_boxes = np.reshape(final_boxes / ratio, (-1, 4))
                dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1),
                                       np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
            else:
                predictions = np.reshape(data, (1, -1, int(5 + self.n_classes)))[0]
                dets = self.postprocess(predictions, ratio)

            if dets is not None:
                final_boxes, final_scores, final_cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
                frame = vis(frame, final_boxes, final_scores, final_cls_inds,
                            conf=conf, class_names=self.class_names)
            cv2.imshow('frame', frame)
            out.write(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        out.release()
        cap.release()
        cv2.destroyAllWindows()

    
    @staticmethod
    def postprocess(predictions, ratio):
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=0.45, score_thr=0.1)
        return dets
    
    def inference(self, img_path, conf=0.5, end2end=False):
        origin_img = cv2.imread(img_path)
        img, ratio = preproc(origin_img, self.imgsz, self.mean, self.std)
        data = self.infer(img)
        if end2end:
            num, final_boxes, final_scores, final_cls_inds = data
            final_boxes = np.reshape(final_boxes/ratio, (-1, 4))
            dets = np.concatenate([final_boxes[:num[0]], np.array(final_scores)[:num[0]].reshape(-1, 1), np.array(final_cls_inds)[:num[0]].reshape(-1, 1)], axis=-1)
        else:
            predictions = np.reshape(data, (1, -1, int(5+self.n_classes)))[0]
            dets = self.postprocess(predictions,ratio)

        if dets is not None:
            final_boxes, final_scores, final_cls_inds = dets[:,
                                                             :4], dets[:, 4], dets[:, 5]
            origin_img = vis(origin_img, final_boxes, final_scores, final_cls_inds,
                             conf=conf, class_names=self.class_names)
        return origin_img

    def __del__(self):
        """
        Ressourcen freigeben (Kontext, Stream, Speicher).
        """
        try:
            # Speicher für Eingaben und Ausgaben freigeben
            if hasattr(self, 'inputs'):
                for inp in self.inputs:
                    if 'device' in inp and inp['device']:
                        inp['device'].free()
            if hasattr(self, 'outputs'):
                for out in self.outputs:
                    if 'device' in out and out['device']:
                        out['device'].free()

            # Stream freigeben
            if hasattr(self, 'stream') and self.stream:
                self.stream = None

            # Kontext nicht freigeben – wird im Thread behandelt
            print(f"{self.__class__.__name__}: Ressourcen freigegeben.")
        except Exception as e:
            print(f"{self.__class__.__name__}: Fehler beim Freigeben der Ressourcen: {e}")


def preproc(image, input_size, mean, std, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.float32)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def rainbow_fill(size=50):  # Einfacher Farbverlauf
    cmap = plt.get_cmap('jet')
    color_list = []

    for n in range(size):
        color = cmap(n / size)
        color_list.append(color[:3])

    return np.array(color_list)


_COLORS = rainbow_fill(80).astype(np.float32).reshape(-1, 3)

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = np.ones((len(keep), 1)) * cls_ind
                dets = np.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

def vis(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = (_COLORS[cls_id] * 255).astype(np.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if np.mean(_COLORS[cls_id]) > 0.5 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)

        txt_bk_color = (_COLORS[cls_id] * 255 * 0.7).astype(np.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img

class AutoencoderEngine:
      def __init__(self, engine_path):
        """
        Initialisiert die TensorRT-Engine.
        """
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        # TensorRT-Engine laden
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Bindings initialisieren
        self.input_binding_idx = self.engine.get_binding_index("input")
        self.output_binding_idx = self.engine.get_binding_index("output")

        # Eingabe- und Ausgabeformen bestimmen
        self.input_shape = self.engine.get_binding_shape(self.input_binding_idx)
        self.output_shape = self.engine.get_binding_shape(self.output_binding_idx)

        print(f"TRTModel: TensorRT-Engine {engine_path} erfolgreich geladen.")

      def infer(self, input_image):
        """
        Führt eine Inferenz mit der TensorRT-Engine aus.
        """
        # Eingabedaten vorbereiten
        input_image = np.ascontiguousarray(input_image.astype(np.float32))
        output = np.empty(self.output_shape, dtype=np.float32)

        # Speicher allokieren
        d_input = cuda.mem_alloc(input_image.nbytes)
        d_output = cuda.mem_alloc(output.nbytes)

        try:
            # Daten in den GPU-Speicher kopieren
            cuda.memcpy_htod(d_input, input_image)
            start_time = time.time()
            self.context.execute_v2([int(d_input), int(d_output)])  # Inferenz ausführen
            cuda.memcpy_dtoh(output, d_output)  # Ausgabe zurückholen
            end_time = time.time()

            # Inferenzzeit berechnen
            inference_time = (end_time - start_time) * 1000
            return output, inference_time
        finally:
            # Speicher freigeben
            d_input.free()
            d_output.free()

    #   def __del__(self):
    #     """
    #     Ressourcen freigeben.
    #     """
    #     try:
    #         # Speicher für Eingaben und Ausgaben wird in der infer-Methode behandelt

    #         # Stream freigeben
    #         if hasattr(self, 'stream') and self.stream:
    #             self.stream = None

    #         # Kontext nicht freigeben – wird im Thread behandelt
    #         print(f"{self.__class__.__name__}: Ressourcen freigegeben.")
    #     except Exception as e:
    #         print(f"{self.__class__.__name__}: Fehler beim Freigeben der Ressourcen: {e}")


def cuda_init():
    # Manuelle CUDA-Kontextinitialisierung
    cuda.init()
    device = cuda.Device(0)  # Erster GPU-Device
    context = device.make_context()  # Kontext für diesen Thread erstellen
    return context





