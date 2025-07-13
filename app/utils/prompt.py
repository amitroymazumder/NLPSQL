# utils/prompt.py

def format_prompt(nl_query, schema_subset):
    table_map = {}
    for table, column in schema_subset:
        table_map.setdefault(table, []).append(column)

    schema_text = "\n".join(
        f"- Table: {table}\n  - " + ", ".join(columns)
        for table, columns in table_map.items()
    )

    examples = """
Example:
User: Show total notional per product in the last 30 days
Response:
{
  "sql": "SELECT product, SUM(notional) as total_notional FROM trades WHERE trade_date >= DATEADD(day, -30, GETDATE()) GROUP BY product",
  "chartConfig": {
    "chartType": "bar",
    "xAxis": "product",
    "yAxis": "total_notional",
    "title": "Total Notional by Product (Last 30 Days)"
  }
}
"""

    return f"""
You are a SQL assistant.
Given the following user request and database schema, return a JSON object with:
1. 'sql': the SQL query to retrieve the correct result
2. 'chartConfig': an object with 'chartType', 'xAxis', 'yAxis', and 'title'

Schema:
{schema_text}

{examples}

User query: {nl_query}
"""
