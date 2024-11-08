import os
import time
import asyncio
import logging
from typing import List, Tuple 
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import onnxruntime as ort
import psutil
from v1.bbox import Bbox


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PersonDetector:
    def __init__(self):
        """
        Initialize the PersonDetector with a YOLOv8 ONNX model.
        """
        self.model_path = os.getenv('MODEL_PATH', 'person_detection_service/model/yolov8s.onnx')
        self.session = ort.InferenceSession(self.model_path)

    async def preprocess_images(self, bytearray_images: List[bytes], target_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, List[float]]:
        async def resize_image_with_padding(image: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
            h, w = image.shape[:2]
            scale = min(target_size[0] / h, target_size[1] / w)
            nh, nw = int(h * scale), int(w * scale)
            image_resized = cv2.resize(image, (nw, nh))
            new_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
            new_image[:nh, :nw] = image_resized
            return new_image, scale, (nw, nh)

        images = [cv2.cvtColor(np.array(Image.open(BytesIO(img))), cv2.COLOR_RGB2BGR) for img in bytearray_images]
        
        resized_images = []
        scales = []
        for img in images:
            resized_img, scale, _ = await resize_image_with_padding(img, target_size)
            resized_images.append(resized_img)
            scales.append(scale)

        input_tensor = np.stack(resized_images).astype(np.float32) / 255.0
        input_tensor = input_tensor.transpose(0, 3, 1, 2)

        return input_tensor, scales

    async def postprocess_person_detection(self, outputs, scales, confidence_threshold=0.4, person_class_id=0, iou_threshold=0.6):
        def adjust_Bboxes(Bboxes: List[Bbox], scale: float) -> List[Bbox]:
            return [Bbox(
                xmin=int(box.xmin / scale), ymin=int(box.ymin / scale),
                xmax=int(box.xmax / scale), ymax=int(box.ymax / scale),
                conf=box.conf) for box in Bboxes]
        
        def calculate_iou(box1: Bbox, box2: Bbox) -> float:
            x1, y1 = max(box1.xmin, box2.xmin), max(box1.ymin, box2.ymin)
            x2, y2 = min(box1.xmax, box2.xmax), min(box1.ymax, box2.ymax)
            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            box1_area = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
            box2_area = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)
            return inter_area / (box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0

        def non_max_suppression(Bboxes: List[Bbox], iou_threshold: float) -> List[Bbox]:
            Bboxes = sorted(Bboxes, key=lambda x: x.conf, reverse=True)
            selected_Bboxes = []
            while Bboxes:
                best_Bbox = Bboxes.pop(0)
                selected_Bboxes.append(best_Bbox)
                Bboxes = [box for box in Bboxes if calculate_iou(best_Bbox, box) < iou_threshold]
            return selected_Bboxes

        all_adjusted_Bboxes = []
        for batch_idx in range(outputs[0].shape[0]):

            detections = outputs[0][batch_idx].transpose((1, 0))
            detected_persons = []

            for detection in detections:
                x_center, y_center, width, height, *class_scores = detection
                xmin = int(x_center - width / 2)
                ymin = int(y_center - height / 2)
                xmax = int(x_center + width / 2)
                ymax = int(y_center + height / 2)
                class_scores = detection[4:]
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]

                if class_id == person_class_id and confidence >= confidence_threshold:
                    detected_persons.append(Bbox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, conf=confidence))

            if detected_persons:
                detected_persons = non_max_suppression(detected_persons, iou_threshold)
                detected_persons = adjust_Bboxes(detected_persons, scales[batch_idx])

            all_adjusted_Bboxes.append(detected_persons)

        return all_adjusted_Bboxes

    async def detect_persons(self, bytearray_images: List[bytes]):
        def log_metrics(stage):
            logger.info(f"{stage} - CPU usage: {psutil.cpu_percent()}%, Memory usage: {psutil.virtual_memory().percent}%")
            
        log_metrics("Before preprocessing")
        start_time_preprocess = time.time()
        input_tensor, scales = await self.preprocess_images(bytearray_images)
        logger.info(f"Preprocessing time: {(time.time() - start_time_preprocess) * 1000:.2f} ms")
        log_metrics("After preprocessing")

        log_metrics("Before inference")
        start_time_inference = time.time()
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})
        logger.info(f"Inference time: {(time.time() - start_time_inference) * 1000:.2f} ms")
        log_metrics("After inference")

        log_metrics("Before post-processing")
        start_time_postprocess = time.time()
        detections = await self.postprocess_person_detection(outputs, scales)
        logger.info(f"Post-processing time: {(time.time() - start_time_postprocess) * 1000:.2f} ms")
        log_metrics("After post-processing")

        return detections
