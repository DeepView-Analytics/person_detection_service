import json
from aiokafka import AIOKafkaConsumer
from person_detection_service.redis_clients.RedisManager import RedisManager
from .producer import KafkaProducerService
from ..model.yolov8 import PersonDetector

class KafkaConsumerService:
    def __init__(self, bootstrap_servers, topic, producer):

        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self.consumer = None  # Initialize the consumer as None
        self.detector = PersonDetector()
        self.frames_metadata_manager_client = RedisManager(db=1)
        self.persons_metadata_manager_client = RedisManager(db=2)
        self.frames_data_manager_client = RedisManager(db=0)
        self.producer = producer

    async def start(self):
        self.consumer = AIOKafkaConsumer(
            self.topic,
            bootstrap_servers=self.bootstrap_servers,
            value_deserializer=lambda x: json.loads(x.decode('utf-8'))

        )
        await self.consumer.start()

        try:
            await self.consume_messages()
            
        finally:
            await self.consumer.stop()


    async def consume_messages(self):
        async for message in self.consumer:
            print("There is a message")
            frames_keys = message.value
  
            frame_data = self.frames_data_manager_client.get_many(frames_keys)

            detection = await self.detector.detect_persons(frame_data)

            response = []
            frames_metadata_update = []
            frames_metadata_keys = []
            persons_metadata_keys = []
            persons_metadata_update = []
            for i, frame_key in enumerate(frames_keys):
                persons_ids = []
                detections = []
                frame_metadata_key = f"metadata:{frame_key}"
                frames_metadata_keys.append(frame_metadata_key)
                
                for person in detection[i]:

                    person_key  = f"{frame_key}:{person.person_key}"
                    person.frame_key = frame_key
                    person_metadata_key = f"metadata:{person_key}"

                    detection_tuple = (person_metadata_key,person.bbox)
                    detections.append(detection_tuple)
                    persons_ids.append(person.person_key)
                    
                    persons_metadata_keys.append(person_metadata_key)
                    person_metadata  = person.model_dump()
                    persons_metadata_update.append(person_metadata)

                serialized_detections = json.dumps(persons_ids)
                frames_metadata_update.append(serialized_detections)
                response.append(detections)


            self.frames_metadata_manager_client.update_by_field_many(frames_metadata_keys,"detected_persons",frames_metadata_update)
            self.persons_metadata_manager_client.save_many(persons_metadata_keys,persons_metadata_update)
            await self.producer.send_detection_response(response)
