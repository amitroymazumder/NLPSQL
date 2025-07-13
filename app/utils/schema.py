# utils/schema.py

import pyodbc
import os
import redis
import hashlib
import json
import logging

logger = logging.getLogger(__name__)

# Redis client
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# SQL connection
SQL_SERVER = os.getenv("SQL_SERVER", "localhost")
DB_NAME = os.getenv("DB_NAME", "TradeDB")
USERNAME = os.getenv("DB_USER", "user")
PASSWORD = os.getenv("DB_PASSWORD", "pass")

conn_str = (
    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
    f'SERVER={SQL_SERVER};'
    f'DATABASE={DB_NAME};'
    f'UID={USERNAME};'
    f'PWD={PASSWORD}'
)

def infer_table_keywords(nl_query):
    return [word.strip(',.') for word in nl_query.lower().split() if len(word) > 3]

def get_relevant_schema(keywords=None):
    cache_key = "schema_subset"
    if keywords:
        key_hash = hashlib.md5(",".join(sorted(keywords)).encode()).hexdigest()
        cache_key = f"schema_subset:{key_hash}"

    cached = redis_client.get(cache_key)
    if cached:
        logger.info("Using cached schema subset")
        return json.loads(cached)

    schema = []
    try:
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT TABLE_NAME, COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS")
            for table, column in cursor.fetchall():
                if not keywords or any(k in table.lower() or k in column.lower() for k in keywords):
                    schema.append((table, column))
    except Exception as e:
        logger.error(f"Schema fetch failed: {e}")

    redis_client.set(cache_key, json.dumps(schema), ex=3600)
    return schema
