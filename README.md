# Natural Language to SQL with Visualization (FastAPI + MS SQL)

This project provides a lightweight and scalable backend service for converting natural language queries into SQL, executing them on an MS SQL Server, and returning results along with chart configurations for visualizing.

## 🔧 Tech Stack
- **FastAPI**: Backend API
- **Redis**: Caching for schema and LLM responses
- **MS SQL Server**: Primary data store
- **In-House LLM API**: Converts NL queries to SQL + chart config
- **Next.js + Recharts**: Frontend for UI and chart rendering

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
│       └── index.tsx         # Main input and chart rendering
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
  "query": "Show total notional per product in the last 30 days"
}
```

## 📊 Sample Response
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
Feel free to fork, adapt to your schema, and extend visual options!
