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
