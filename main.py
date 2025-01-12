import os
from fastapi import FastAPI
from person_detection_service.kafka_interaction.consumer import KafkaConsumerService
from person_detection_service.kafka_interaction.producer import KafkaProducerService
import uvicorn
from contextlib import asynccontextmanager

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = os.getenv('KAFKA_BOOTSTRAP_SERVERS', '127.0.0.1:9093')
KAFKA_TOPIC_REQUESTS = os.getenv('KAFKA_TOPIC_REQUESTS', 'person_detection_requests')
KAFKA_TOPIC_RESPONSES = os.getenv('KAFKA_TOPIC_RESPONSES', 'person_detected_response')  




# Initialize Kafka Producer and Consumer
producer = KafkaProducerService(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, topic=KAFKA_TOPIC_RESPONSES)
consumer = KafkaConsumerService(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, topic=KAFKA_TOPIC_REQUESTS)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    await producer.start()
    await consumer.start()
    yield
    # Shutdown logic
    await producer.close()
    await consumer.consumer.stop()  

app = FastAPI(lifespan=lifespan)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Person Detection Service!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv('FASTAPI_PORT', 8000)))
