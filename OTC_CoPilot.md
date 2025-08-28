Got it ✅ — now that we’ve explored Vanna’s core code, the Streamlit app, and your target stack (MS SQL + ChromaDB + in-house LLM), I’ll design a clean project structure for your OTC Ops Copilot.

This structure combines only the essential files from:

Vanna Core (NL → SQL engine logic)

Streamlit UI (user interface)

Your new modules (ChromaDB retriever, in-house LLM connector, YAML knowledge loader)



---

📂 Proposed Project Structure

otc-ops-copilot/
│
├── app/                          # Streamlit frontend
│   ├── app.py                    # Main entrypoint for UI
│   ├── components/               # UI widgets
│   │   ├── query_input.py
│   │   ├── results_table.py
│   │   └── charts.py
│   └── pages/                    # Multi-page support (if needed)
│       ├── dashboards.py
│       └── reports.py
│
├── core/                         # Core logic (adapted from Vanna)
│   ├── __init__.py
│   ├── engine.py                 # Orchestrates NL → SQL pipeline
│   ├── retriever.py              # ChromaDB hierarchical retrieval
│   ├── sql_generator.py          # Calls LLM to generate SQL
│   ├── sql_validator.py          # Ensures SELECT-only, valid joins
│   ├── db_connector.py           # MS SQL read-only connector
│   └── visualizer.py             # Plotly-based charts
│
├── knowledge/                    # Knowledge layer (our YAML + DB sync)
│   ├── knowledge_base.yaml       # Glossary, rules, constraints, examples
│   ├── loader.py                 # Loads YAML into ChromaDB
│   └── sync_masterdata.py        # Optional ETL: syncs reference tables
│
├── services/                     # Optional extensions
│   ├── scheduler.py              # APScheduler for report scheduling
│   ├── report_generator.py       # WeasyPrint/ReportLab for PDF/Excel
│   └── notifier.py               # Email/Slack integration
│
├── tests/                        # Unit tests
│   ├── test_sql_validator.py
│   ├── test_retriever.py
│   └── test_examples.py
│
├── config/                       # Configurations
│   ├── chromadb_config.yaml      # Embedding + persistence settings
│   ├── db_config.yaml            # MS SQL connection details (read-only)
│   └── llm_config.yaml           # In-house LLM connector info
│
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
└── run.py                        # Entry point (launch backend + UI)


---

🔹 File Roles

Frontend (Streamlit)

app/app.py: Main Streamlit entrypoint (query input + results + charts).

components/: Small modular widgets (e.g., text input, charts).

pages/: Optional Streamlit pages for dashboards, saved reports.



---

Core Logic (from Vanna)

engine.py: Pipeline (NL → retrieval → LLM → SQL → results).

retriever.py: Hierarchical ChromaDB retriever (schema + glossary + constraints + examples).

sql_generator.py: Calls your in-house LLM with context → SQL.

sql_validator.py: Enforces read-only queries, validates joins.

db_connector.py: Connects to MS SQL (read-only replica).

visualizer.py: Converts results → Plotly visualizations.



---

Knowledge Layer

knowledge_base.yaml: Master glossary, rules, reference data, example queries.

loader.py: Loads YAML into ChromaDB (embedding + metadata).

sync_masterdata.py: Optional job to keep reference tables synced.



---

Services (Extensions)

scheduler.py: Allows ops to schedule queries/reports.

report_generator.py: Converts query results → PDF/Excel.

notifier.py: Sends reports via email/Slack.



---

Config

db_config.yaml: DB connection details (use read-only credentials).

llm_config.yaml: How to connect to in-house LLM (API endpoint, keys).

chromadb_config.yaml: Path & persistence options for ChromaDB.



---

Run & Deployment

requirements.txt: Python dependencies (Streamlit, ChromaDB, pyodbc, Plotly, etc.).

run.py: CLI entrypoint (python run.py) to:

Load knowledge base → ChromaDB

Start backend engine

Launch Streamlit frontend




---

✅ This gives you a lean, production-ready structure.
Everything unnecessary from Vanna’s repo (extras, connectors you don’t need, docs, etc.) is stripped away.


---

👉 Do you want me to also create a requirements.txt draft with only the needed dependencies for this cleaned-up version?

