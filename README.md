# Natural Language to SQL with Visualization (FastAPI + MS SQL)

This project provides a lightweight and scalable backend service for converting natural language queries into SQL, executing them on an MS SQL Server, and returning results along with chart configurations.

## 🔧 Tech Stack
- **FastAPI**: Web framework
- **Redis**: Caching LLM responses and schema metadata
- **MS SQL Server**: Data source
- **In-House LLM API**: Converts NL queries to SQL + Chart config
- **Recharts (Frontend)**: Visualizes the chart based on config

## 📁 Project Structure
```
/app
├── main.py              # FastAPI app setup and router registration
├── models.py            # Request models
├── routes/
│   └── nl_to_sql.py     # Main API endpoint
├── utils/
│   ├── schema.py        # Keyword inference + schema subset from DB
│   ├── prompt.py        # Prompt construction for LLM
│   └── validate.py      # SQL safety checks
```

## 🧠 Features
- Schema subsetting to reduce LLM hallucination
- Redis caching for speed and efficiency
- SQL validation (only allows `SELECT` queries)
- Returns chart config (type, axis, title) for visualizing results

## 🚀 Getting Started

1. Clone this repo:
```bash
git clone https://github.com/your-org/nl-to-sql-api.git
cd nl-to-sql-api
```

2. Install dependencies:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Set environment variables:
```
REDIS_URL=redis://localhost:6379/0
SQL_SERVER=localhost
DB_NAME=YourDB
DB_USER=YourUser
DB_PASSWORD=YourPassword
LLM_API_URL=http://localhost:8001/generate
```

4. Run the server:
```bash
uvicorn main:app --reload
```

## 🧪 Example Request
```json
{
  "query": "Show total notional per product in the last 30 days"
}
```

## 📈 Response
```json
{
  "data": [...],
  "chartConfig": {
    "chartType": "bar",
    "xAxis": "product",
    "yAxis": "total_notional",
    "title": "Total Notional by Product"
  }
}
```

---

Feel free to fork and customize based on your internal schema and LLM integrations.
