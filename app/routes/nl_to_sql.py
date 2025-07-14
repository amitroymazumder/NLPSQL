# routes/nl_to_sql.py

from fastapi import APIRouter, HTTPException
from app.models import QueryRequest
from app.utils.schema import get_relevant_schema
from app.utils.prompt import build_sql_prompt, build_summary_prompt
from app.utils.validate import is_safe_sql
import pyodbc
import requests
import os
import json

router = APIRouter()

SQL_SERVER = os.getenv("SQL_SERVER")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
LLM_API_URL = os.getenv("LLM_API_URL")

@router.post("/api/nl-to-sql")
async def nl_to_sql(req: QueryRequest):
    try:
        # Step 1: Schema narrowing
        schema_subset = get_relevant_schema(req.query)

        # Step 2: SQL + chart prompt
        prompt = build_sql_prompt(req.query, schema_subset)
        llm_response = requests.post(LLM_API_URL, json={"prompt": prompt})
        llm_response.raise_for_status()
        result = llm_response.json()

        sql = result.get("sql")
        chart_config = result.get("chartConfig")

        if not sql or not is_safe_sql(sql):
            raise HTTPException(status_code=400, detail="Invalid or unsafe SQL generated")

        # Step 3: Execute SQL
        conn_str = (
            f"DRIVER={{ODBC Driver 17 for SQL Server}};"
            f"SERVER={SQL_SERVER};DATABASE={DB_NAME};UID={DB_USER};PWD={DB_PASSWORD}"
        )
        conn = pyodbc.connect(conn_str)
        cursor = conn.cursor()
        cursor.execute(sql)
        columns = [column[0] for column in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
        cursor.close()
        conn.close()

        # Step 4: Summary generation
        summary_prompt = build_summary_prompt(req.query, rows)
        summary_response = requests.post(LLM_API_URL, json={"prompt": summary_prompt})
        summary_response.raise_for_status()
        summary = summary_response.json().get("text")

        return {
            "sql": sql,
            "data": rows,
            "chartConfig": chart_config,
            "summary": summary
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
