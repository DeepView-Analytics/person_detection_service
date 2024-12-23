import json
import os
import redis
import time
from typing import List, Dict , Any

class RedisManager:
    def __init__(self, db=0, max_retries=5, backoff_factor=0.1):
        self.client = redis.StrictRedis(
            host=os.getenv('REDIS_SERVER', 'localhost'), 
            port=os.getenv('REDIS_PORT', 6379), 
            db=db
        )
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def save_one(self, key: str, value: Dict[str, Any]):
        self.client.hset(key, mapping=value)

    def get_one(self, key: str, field: str):
        return self.client.hgetall(key)

    def save_many(self, keys: List[str], values: List[Dict[str, Any]]):
        with self.client.pipeline() as pipe:
            for key, value in zip(keys, values):
                # Serialize the dictionary to a JSON string
                serialized_value = {field: json.dumps(v) for field, v in value.items()}
                # Use HSET with serialized fields and values
                for field, val in serialized_value.items():
                    pipe.hset(key, field, val)
            pipe.execute()

    def get_many(self, keys: List[str]):
        with self.client.pipeline() as pipe:
            for key in keys:
                pipe.get(key)
            return pipe.execute()

    def update_by_field_one(self, key: str, field: str, value):
        lua_script = """
        local key = KEYS[1]
        local field = ARGV[1]
        local value = ARGV[2]
        
        redis.call('HSET', key, field, value)
        return redis.call('HGET', key, field)
        """
        script = self.client.register_script(lua_script)
        for attempt in range(self.max_retries):
            try:
                return script(keys=[key], args=[field, value])
            except redis.exceptions.WatchError:
                time.sleep(self.backoff_factor * (2 ** attempt))  # Exponential backoff
        raise Exception(f"Failed to update field '{field}' in key '{key}' after {self.max_retries} attempts")

    def update_by_field_many(self, keys: List[str], field: str, values: List[str]):
        db_index = self.client.connection_pool.connection_kwargs.get("db", 5)
        print(f"Current Redis database: {db_index}")
        for i ,key in enumerate(keys):
            print(type(key))
            print(len(key))
            print(f"-{key}-")
            print(f"Does 'metadatakey' exist? {self.client.exists(key)}")
            ttl = self.client.ttl(key)
            print(f"TTL for 'metadatakey': {ttl}")
            key_type = self.client.type(key)
            print(f"The type of 'key' {i} is: {key_type.decode()}")  # Output: hash
        """
        Update a specific field in multiple Redis hash keys.
        :param keys: List of Redis hash keys.
        :param field: The field to update.
        :param values: List of values to set for the field.
        """
        # Lua script for updating multiple fields
        lua_script = """
        for i, key in ipairs(KEYS) do
            redis.call('HSET', key, ARGV[1], ARGV[i+1])
        end
        return true
        """
        # Validate inputs
        if not all(isinstance(key, str) for key in keys):
            raise ValueError("All keys must be strings.")
        if not isinstance(field, str):
            raise ValueError("Field must be a string.")
        if not all(isinstance(value, str) for value in values):
            raise ValueError("All values must be strings.")
        if len(values) != len(keys):
            raise ValueError("The number of values must match the number of keys.")

        # Retry mechanism
        for attempt in range(self.max_retries):
            try:
                # Attempt to execute the Lua script
                self.client.eval(lua_script, len(keys), *(keys + [field] + values))
                return  # Exit if the script executes successfully
            except redis.exceptions.WatchError:
                time.sleep(self.backoff_factor * (2 ** attempt))  # Exponential backoff
            except redis.exceptions.RedisError as e:
                print(f"Redis error during update: {e}")
                raise
        # If retries are exhausted
        raise Exception(f"Failed to update field '{field}' in keys '{keys}' after {self.max_retries} attempts")



    def get_values_of_field_many(self, keys: List[str], field: str) -> List[str]:
        lua_script = """
        local result = {}
        for i, key in ipairs(KEYS) do
            result[i] = redis.call('HGET', key, ARGV[1])
        end
        return result
        """
        script = self.client.register_script(lua_script)
        results = script(keys=keys, args=[field])
        return [result.decode('utf-8') for result in results if result]

