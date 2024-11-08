import json
import os
from aiokafka import AIOKafkaConsumer

import redis
from v1.detectionrequest import DetectionRequest 
from v1.objectdetected import ObjectDetected
from .producer import KafkaProducerService
from ..model.yolov8 import PersonDetector

class KafkaConsumerService:
    def __init__(self, bootstrap_servers='192.168.111.131:9092', topic='person_detection_requests'):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.consumer = None  # Initialize the consumer as None
        self.detector = PersonDetector()
        self.producer = KafkaProducerService()
        self.redis_client = redis.StrictRedis(
            host=os.getenv('REDIS_SERVER', 'localhost'), 
            port=os.getenv('REDIS_PORT', 6379), 
            db=os.getenv('REDIS_DB_ID', 0)
        )

    async def start(self):
        # Initialize the consumer
        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda x: x.decode('utf-8')

        )
        await self.consumer.start()

        # Start the producer
        await self.producer.start()

        try:
            await self.consume_messages()
        finally:
            await self.consumer.stop()
            await self.producer.close()

    async def consume_messages(self):
        async for message in self.consumer:
            print("There is a message")
            request = json.loads(message.value)
            detection_request = DetectionRequest(**request)
            keys = [f"{frame.camera_id}:{frame.timestamp}" for frame in detection_request.frames]
            frame_data = self.redis_client.mget(keys)

            # Run detection asynchronously
            detection = await self.detector.detect_persons(frame_data)
            results = []
            
            for i, frame in enumerate(detection_request.frames):
                results.append(ObjectDetected(camera_id=frame.camera_id, object_detected=detection[i], timestamp=frame.timestamp))
            
            # Send response to Kafka
            await self.producer.send_detection_response(detection_request.request_id, results)
