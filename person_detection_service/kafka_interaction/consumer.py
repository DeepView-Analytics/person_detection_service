import json
import os
from threading import Thread
from kafka import KafkaConsumer
import redis
from v1.detectionrequest import DetectionRequest 
from v1.objectdetected import ObjectDetected
from .producer import KafkaProducerService
from ..model.yolov8 import PersonDetector

class KafkaConsumerService:
    def __init__(self, bootstrap_servers='192.168.111.131:9092', topic='person_detection_requests'):
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_servers,
            api_version = tuple(map(int, os.getenv('KAFKA_API_VERSION').split(','))),
            value_deserializer=lambda x: x.decode('utf-8')
        )
        self.detector = PersonDetector()
        self.running = True
        self.producer = KafkaProducerService()
        self.redis_client = redis.StrictRedis(host=os.getenv('REDIS_SERVER'), port=os.getenv('REDIS_PORT'), db=os.getenv('REDIS_DB_ID'))

    def start(self):
        Thread(target=self.consume_messages, daemon=True).start()

    def consume_messages(self):
        try:
           
            for message in self.consumer:
                if not self.running:
                    break
                print("there is a message")
                request = message.value
                request = json.loads(request)
                detection_request = DetectionRequest(**request)
                keys = [f"{frame.camera_id}:{frame.timestamp}" for frame in detection_request.frames]
                frame_data = self.redis_client.mget(keys)
                detection = self.detector.detect_persons(frame_data)
                results = []
                for i , frame in enumerate(detection_request.frames):
                    results.append(ObjectDetected(camera_id=frame.camera_id, object_detected=detection[i], timestamp=frame.timestamp))
                self.producer.send_detection_response(detection_request.request_id,results)
                    
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.consumer.close()  # Ensure the consumer closes properly

    def stop(self):
        self.running = False
        self.consumer.close()
