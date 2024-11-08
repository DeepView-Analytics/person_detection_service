import json
import logging
import os
from aiokafka import AIOKafkaProducer
from v1.detectionresponse import DetectionResponse
from typing import List, Tuple
from math import ceil
from v1.partitioneddetectionbatch import PartitionedDetectionBatch
from v1.objectdetected import ObjectDetected

class KafkaProducerService:
    def __init__(self, bootstrap_servers='127.0.0.1:9092', topic='person_detected_response'):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None  # Initialize the producer as None
        self.max_bboxes_per_batch = int(os.getenv('RESPONSE_BATCH_SIZE', 10))


    async def start(self):
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        await self.producer.start()





    def balance_batches(self , detections: List[ObjectDetected], max_bboxes_per_batch ):
        pile = [(detection, 1) for detection in detections]
  
        grouped_batches = []
        free_space = max_bboxes_per_batch
        current_batch = []
        while pile:
            current_item, returning_times = pile.pop(0)
            detections = current_item.object_detected

            if len(detections) < free_space:
                partitioned_batch = PartitionedDetectionBatch(
                    camera_id=current_item.camera_id,
                    timestamp=current_item.timestamp,
                    object_detected=detections,
                    partition_number=returning_times ,
                    total_partitions=returning_times  
                )

                current_batch.append(partitioned_batch)
                free_space -= len(detections)
            else:
                remaining_detections = detections[free_space:]
                partitioned_batch = PartitionedDetectionBatch(
                    camera_id=current_item.camera_id,
                    timestamp=current_item.timestamp,
                    object_detected=detections[:free_space],
                    partition_number=returning_times ,
                    total_partitions=ceil(len(remaining_detections) / max_bboxes_per_batch) + returning_times
                )
                current_batch.append(partitioned_batch)

                grouped_batches.append(current_batch)
                current_batch = []

                if remaining_detections:
                    pile.insert(0, (ObjectDetected(
                        camera_id=current_item.camera_id,
                        timestamp=current_item.timestamp,
                        object_detected=remaining_detections
                    ), returning_times +1))
                free_space = max_bboxes_per_batch

        if current_batch :
            grouped_batches.append(current_batch)
        print(f"len of batchs to send : {len(grouped_batches)}")
        return grouped_batches



    async def send_detection_response(self, request_id, detections):

        print("Trying to cast on DetectionResponse")
        batchs = self.balance_batches(detections,max_bboxes_per_batch=self.max_bboxes_per_batch)
        print(len(batchs))
        for batch in batchs : 

            response_message = DetectionResponse(request_id=request_id, detection=batch)
            print("Casted on DetectionResponse!")
            response_message = response_message.model_dump()

            try:
                print(f"The topic target is: {self.topic}")
                future = await self.producer.send_and_wait(self.topic, response_message)
                print(f"Message sent successfully to topic {self.topic}")
            except Exception as e:
                logging.error(f"Failed to send message: {e}", exc_info=True)

    async def close(self):
        await self.producer.stop()