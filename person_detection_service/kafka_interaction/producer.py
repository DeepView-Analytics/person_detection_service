import json
import logging
import os
from kafka import KafkaProducer
from v1.detectionresponse import DetectionResponse



class KafkaProducerService:
    def __init__(self, bootstrap_servers='127.0.0.1:9092', topic='person_detected_response'):
        self.producer = KafkaProducer(
            bootstrap_servers=bootstrap_servers,
            api_version=tuple(map(int, os.getenv('KAFKA_API_VERSION').split(','))),
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            request_timeout_ms=30000
        )
        self.topic = topic

    def send_detection_response(self, request_id, detections):
        print("trying to cast on DetectionResponse")
      
        response_message = DetectionResponse(request_id=request_id, detection=detections)
        
        print(" casted  on DetectionResponse !!")
        response_message = response_message.model_dump()

        message_size = len(str(response_message))
        print(f"Message size: {message_size} bytes")
        print("Attempting to send message to topic")
         # Debugging print for final object schema
        try:
            print(f"the topic target is : {self.topic}")
            future = self.producer.send(self.topic, response_message)
            record_metadata = future.get(timeout=10)
            print(f"Message sent successfully to topic {record_metadata.topic} partition {record_metadata.partition} at offset {record_metadata.offset}")
        except Exception as e:
            logging.error(f"Failed to send message: {e}", exc_info=True)
        finally:
            self.producer.flush()

    def close(self):
        self.producer.flush()
        self.producer.close()