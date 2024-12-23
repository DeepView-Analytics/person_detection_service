import json
import logging
import operator
import os
from aiokafka import AIOKafkaProducer
from typing import List, Tuple
from math import ceil
from v3.partitioneddetectionbatch import PartitionedDetectionBatch


class KafkaProducerService:
    def __init__(self, bootstrap_servers='127.0.0.1:9092', topic='person_detected_response'):
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.producer = None  # Initialize the producer as None
        self.max_bboxes_per_batch = int(os.getenv('RESPONSE_BATCH_SIZE', 10))


    async def start(self):
        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: v.encode('utf-8')
            )
            await self.producer.start()
        except Exception as e:
            logging.error(f"Failed to start Kafka producer: {e}", exc_info=True)
            raise e





    def balance_batches(self , detections: List[List[Tuple]], max_bboxes_per_batch ):
        

        pile = [(detection, 1) for detection in detections]
  
        grouped_batches = []
        free_space = max_bboxes_per_batch
        current_batch = []
        while pile:
            current_item, returning_times = pile.pop(0)
            detections = current_item
        
            if len(detections) < free_space:
                keys = list(map(operator.itemgetter(0), detections))
                dets = list(map(operator.itemgetter(1), detections))
                partitioned_batch = PartitionedDetectionBatch(
                    frame_key =  ":".join(keys[0].split(":", 3)[1:3]),
                    person_keys = keys,
                    person_bbox = dets,
                    partition_number=returning_times ,
                    total_partitions=returning_times  
                )

                current_batch.append(partitioned_batch)
                free_space -= len(detections)
            else:
                remaining_detections = detections[free_space:]
                keys = list(map(operator.itemgetter(0), detections[:free_space]))
                dets = list(map(operator.itemgetter(1), detections[:free_space]))
                partitioned_batch = PartitionedDetectionBatch(
                    frame_key =  ":".join(keys[0].split(":", 3)[1:3]),
                    person_keys = keys,
                    person_bbox = dets,
                    partition_number=returning_times ,
                    total_partitions=ceil(len(remaining_detections) / max_bboxes_per_batch) + returning_times
                )
                current_batch.append(partitioned_batch)

                grouped_batches.append(current_batch)
                current_batch = []

                if remaining_detections:
                    pile.insert(0, (remaining_detections, returning_times +1))
                free_space = max_bboxes_per_batch

        if current_batch :
            grouped_batches.append(current_batch)
  
        return grouped_batches



    async def send_detection_response(self, detections):
        batchs = self.balance_batches(detections,max_bboxes_per_batch=self.max_bboxes_per_batch)
        print(f"len batchs = {len(batchs)}")
        for batch in batchs : 
            print(len(batch))
            response_message = json.dumps([partition.model_dump() for partition in batch])
            try:
                print(f"The topic target is: {self.topic}")
                future = await self.producer.send_and_wait(self.topic, response_message)
                print(f"Message sent successfully to topic {self.topic}")
            except Exception as e:
                logging.error(f"Failed to send message: {e}", exc_info=True)

    async def close(self):
        await self.producer.stop()