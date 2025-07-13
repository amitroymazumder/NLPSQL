# routes/nl_to_sql.py

from fastapi import APIRouter
from models import QueryRequest
from utils.schema import infer_table_keywords, get_relevant_schema
from utils.prompt import format_prompt
from utils.validate import is_safe_sql
import logging
import hashlib
import os
import json
import requests
import pyodbc
import redis

router = APIRouter()

logger = logging.getLogger(__name__)

# Redis setup
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)

# SQL Server credentials
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

LLM_API_URL = os.getenv("LLM_API_URL", "http://localhost:8001/generate")

@router.post("/api/nl-to-sql")
async def natural_language_to_sql(request: QueryRequest):
    logger.info(f"Received NL query: {request.query}")

    keywords = infer_table_keywords(request.query)
    schema_subset = get_relevant_schema(keywords)
    prompt = format_prompt(request.query, schema_subset)

    cache_key = f"llm_response:{hashlib.md5(prompt.encode()).hexdigest()}"
    cached_response = redis_client.get(cache_key)

    if cached_response:
        logger.info("Using cached LLM response")
        parsed_json = json.loads(cached_response)
    else:
        try:
            logger.info("Calling LLM API...")
            response = requests.post(LLM_API_URL, json={"prompt": prompt})
            response.raise_for_status()
            parsed_json = response.json()
            redis_client.set(cache_key, json.dumps(parsed_json), ex=600)
        except Exception as e:
            logger.error(f"LLM request failed: {str(e)}")
            return {"error": f"LLM request failed: {str(e)}"}

    sql = parsed_json.get("sql")
    chart_config = parsed_json.get("chartConfig")

    logger.info(f"LLM Response SQL: {sql}")
    logger.info(f"Chart Config: {chart_config}")

    if not is_safe_sql(sql):
        logger.warning("Unsafe SQL detected. Blocking execution.")
        return {"error": "Only SELECT queries are allowed."}

    try:
        logger.info("Executing SQL...")
        with pyodbc.connect(conn_str) as conn:
            cursor = conn.cursor()
            cursor.execute(sql)
            columns = [column[0] for column in cursor.description]
            rows = cursor.fetchall()
            data = [dict(zip(columns, row)) for row in rows]
    except Exception as e:
        logger.error(f"SQL execution failed: {str(e)}")
        return {"error": f"SQL execution failed: {str(e)}"}

    return {
        "data": data,
        "chartConfig": chart_config
    }
