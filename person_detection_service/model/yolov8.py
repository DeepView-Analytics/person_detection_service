import os
import time
from typing import List, Tuple 
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import onnxruntime as ort
import psutil
import redis


from v1.bbox import Bbox
from v1.frame import Frame
from v1.objectdetected import ObjectDetected


class PersonDetector:
    def __init__(self):
        """
        Initialize the PersonDetector with a YOLOv8 ONNX model.
        """
        self.model_path = os.getenv('MODEL_PATH')
        self.session = ort.InferenceSession(self.model_path)

    def preprocess_images(self , bytearray_images: List[bytes], target_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, List[float]]:
    # Convert bytearray images to numpy arrays
        def resize_image_with_padding(image: np.ndarray, target_size: Tuple[int, int] = (640, 640)) -> Tuple[np.ndarray, float, Tuple[int, int]]:
            h, w = image.shape[:2]
            scale = min(target_size[0] / h, target_size[1] / w)
            nh, nw = int(h * scale), int(w * scale)
            image_resized = cv2.resize(image, (nw, nh))
            new_image = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
            new_image[:nh, :nw] = image_resized
            return new_image, scale, (nw, nh)


        images = [cv2.cvtColor(np.array(Image.open(BytesIO(img))), cv2.COLOR_RGB2BGR) for img in bytearray_images]
        
        # Resize images and keep track of scales
        resized_images = []
        scales = []
        for img in images:
            resized_img, scale, _ = resize_image_with_padding(img, target_size)
            resized_images.append(resized_img)
            scales.append(scale)

        # Convert images to tensor format (N, C, H, W) and normalize to [0, 1]
        input_tensor = np.stack(resized_images).astype(np.float32) / 255.0  # Normalize to [0, 1]
        input_tensor = input_tensor.transpose(0, 3, 1, 2)  # Change to CHW format


        return input_tensor, scales
    

    def postprocess_person_detection(self, outputs, scales, confidence_threshold=0.7, person_class_id=0, iou_threshold=0.6):
        """
        Post-process the output from the ONNX model to extract and adjust person bounding boxes for each image in the batch.
        
        Args:
            outputs (list): Model output tensor, shape (batch_size, 84, 8400).
            scales (list): Scale factors for each image in the batch.
            confidence_threshold (float): Minimum confidence score to consider a detection valid.
            person_class_id (int): Class ID for 'person' in the model's output classes.
            iou_threshold (float): Intersection over Union (IoU) threshold for non-maximum suppression.
        
        Returns:
            List of lists containing adjusted bounding boxes for each image in the batch.
        """
        
        def adjust_Bboxes(Bboxes: List[Bbox], scale: float) -> List[Bbox]:
            adjusted_Bboxes = []
            for box in Bboxes:
                xmin = int(box.xmin / scale)
                ymin = int(box.ymin / scale)
                xmax = int(box.xmax / scale)
                ymax = int(box.ymax / scale)
                conf = box.conf
                adjusted_Bboxes.append(Bbox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, conf=conf))
            return adjusted_Bboxes
        
        def calculate_iou(box1: Bbox, box2: Bbox) -> float:
            """Calculate the Intersection over Union (IoU) between two bounding boxes."""
            x1 = max(box1.xmin, box2.xmin)  # xmin
            y1 = max(box1.ymin, box2.ymin)  # ymin
            x2 = min(box1.xmax, box2.xmax)  # xmax
            y2 = min(box1.ymax, box2.ymax)  # ymax

            inter_area = max(0, x2 - x1) * max(0, y2 - y1)
            box1_area = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin)
            box2_area = (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin)

            iou = inter_area / (box1_area + box2_area - inter_area) if (box1_area + box2_area - inter_area) > 0 else 0
        
            return iou

        # Non-Maximum Suppression
        def non_max_suppression(Bboxes: List[Bbox], iou_threshold: float) -> List[Bbox]:
            """Perform Non-Maximum Suppression (NMS) on a list of bounding boxes."""
            if len(Bboxes) == 0:
                return []
            
            # Sort the bounding boxes by confidence score in descending order
            Bboxes = sorted(Bboxes, key=lambda x: x.conf, reverse=True)
            selected_Bboxes: List[Bbox] = []


            while Bboxes:

                best_Bbox = Bboxes.pop(0)  # Get the highest confidence Bbox
                selected_Bboxes.append(best_Bbox)  # Keep it
                
                # Remove boxes with IoU > threshold
                Bboxes = [Bbox for Bbox in Bboxes if calculate_iou(best_Bbox, Bbox) < iou_threshold]

            return selected_Bboxes

        all_adjusted_Bboxes: List[Bbox] = []
    
        for batch_idx in range(outputs[0].shape[0]):
            detections = outputs[0][batch_idx]  # Shape: (84, 8400)
            detected_persons : List[Bbox] = []

            detections = detections.transpose((1, 0))  # Transpose to (8400, 84) for easier processing

            for detection in detections:
                xmax, ymin, xmin, ymax = detection[:4]
                if xmax <= xmin :
                    t = xmax
                    xmax = xmin
                    xmin = t
                elif ymax <= ymin :
                    t = xmax
                    ymax = ymin
                    ymin = t
                class_scores = detection[4:]
                class_id = np.argmax(class_scores)
                confidence = class_scores[class_id]
                if class_id == person_class_id and confidence >= confidence_threshold:
                    detected_persons.append(
            Bbox(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax, conf=confidence)  # Create Bbox object
        )

            if detected_persons:

                detected_persons = non_max_suppression(detected_persons,iou_threshold)  

                detected_persons = adjust_Bboxes(detected_persons, scales[batch_idx])
            all_adjusted_Bboxes.append(detected_persons)

        return all_adjusted_Bboxes

    

    def detect_persons(self, bytearray_images: List[bytes]) :
        """
        Process a batch of frames and detect persons in each frame.

        Args:
            frames: A list of Frame objects containing camera_id and the actual frame data.

        Returns:
            A list of ObjectDetected instances mapping camera_id to detected objects.
        """
        def log_metrics(stage):
            print(f"{stage} - CPU usage: {psutil.cpu_percent()}%, "
                f"Memory usage: {psutil.virtual_memory().percent}%")
            

    


        # Preprocessing images
        log_metrics("Before preprocessing")
        start_time_preprocess = time.time()
        input_tensor, scales = self.preprocess_images(bytearray_images)
        end_time_preprocess = time.time()
        print(f"Preprocessing time: {(end_time_preprocess - start_time_preprocess) * 1000:.2f} ms")
        log_metrics("After preprocessing")

        # Inference
        log_metrics("Before inference")
        start_time_inference = time.time()
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: input_tensor})
        end_time_inference = time.time()
        print(f"Inference time: {(end_time_inference - start_time_inference) * 1000:.2f} ms")
        log_metrics("After inference")

        # Post-processing
        log_metrics("Before post-processing")
        start_time_postprocess = time.time()
        detections = self.postprocess_person_detection(outputs, scales)
        end_time_postprocess = time.time()
        print(f"Post-processing time: {(end_time_postprocess - start_time_postprocess) * 1000:.2f} ms")
        log_metrics("After post-processing")

       
        return detections