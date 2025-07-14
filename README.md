# Natural Language to SQL with Visualization (FastAPI + MS SQL)

This project provides a lightweight and scalable full-stack app for converting natural language queries into SQL, executing them on an MS SQL Server, visualizing the result as a chart, and explaining the data in plain English.

## 🔧 Tech Stack
- **FastAPI**: Backend API
- **Redis**: Caching for schema and LLM responses
- **MS SQL Server**: Primary data store
- **In-House LLM API**: Converts NL queries to SQL + chart config + summary
- **Next.js + Recharts**: Frontend UI for query input, data table, and chart rendering

## 📁 Project Structure
```
nl-to-sql-api/
├── .env.example              # Sample env vars
├── .gitignore
├── README.md
├── requirements.txt          # Backend dependencies
├── app/
│   ├── main.py               # FastAPI app setup
│   ├── models.py             # Pydantic input model
│   ├── routes/
│   │   └── nl_to_sql.py      # Endpoint: POST /api/nl-to-sql
│   └── utils/
│       ├── schema.py         # Schema lookup and Redis caching
│       ├── prompt.py         # Prompt construction logic
│       └── validate.py       # SQL safety check
├── frontend/
│   ├── package.json          # Next.js + Recharts + Axios setup
│   └── pages/
│       └── index.tsx         # Main input + chart/table toggle + SQL + summary
```

## 🚀 Getting Started

### Backend Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 🌐 Environment Configuration
Copy `.env.example` to `.env` and update values:
```
SQL_SERVER=localhost
DB_NAME=YourDatabase
DB_USER=YourUsername
DB_PASSWORD=YourPassword
REDIS_URL=redis://localhost:6379/0
LLM_API_URL=http://localhost:8001/generate
```

## 🧪 Sample Request
```json
{
  "query": "Compare unicorn counts in SF and NY over time"
}
```

## 📊 Sample Response
```json
{
  "sql": "SELECT ...",
  "data": [...],
  "chartConfig": {
    "chartType": "line",
    "xAxis": "year",
    "yAxis": "unicorn_count",
    "title": "Unicorn Count in SF vs NY"
  },
  "summary": "San Francisco generally has a higher unicorn count compared to New York..."
}
```

## ✨ Features
- Natural language to SQL conversion using in-house LLM
- Schema subsetting and Redis caching to reduce hallucinations
- Safe SQL validation (only SELECTs allowed)
- Chart + Table toggle in frontend
- Natural language summary generated from results
- SQL query display for transparency
- Recharts integration for interactive graphs

---
Feel free to fork, adapt to your schema, and extend visual options!
