# OTC Ops Copilot

*AI-Powered NL → SQL Querying, Dashboards & Reporting for Trade Confirmation & Pre-Settlement Data*

---

## 🚩 Problem Statement

Operations teams in banks face challenges with:

* Manual SQL/Excel extracts for daily ops tasks
* Complex schemas with 100+ tables, making query writing hard
* Time-consuming reconciliations and exception checks
* Manual preparation of daily/weekly reports
* Delayed insights due to static dashboards

---

## 💡 Solution Overview – OTC Ops Copilot

Our solution provides:

* Natural language to SQL query generation
* Schema + business logic enforced training for accuracy
* Interactive dashboards & visualizations
* Report scheduling & automated delivery
* Audit logs for compliance and traceability
* Plug-and-play deployment for operations teams

---

## ⚙️ Functional Capabilities

* Ask questions in natural language → Get SQL results instantly
* Dashboards that refresh with real-time trade data
* Save queries as widgets for monitoring KPIs
* Schedule reports (daily, weekly, custom) with email/Slack delivery
* Business glossary + rules enforced (STP, fails, netting logic)
* Audit trail of all queries, results, and reports

---

## 🎯 Business Benefits

* 50–70% reduction in manual report preparation time
* Ops teams self-serve insights without SQL knowledge
* Consistency in applying business rules across queries
* Improved decision-making with real-time dashboards
* Compliance-ready with audit logs
* Scalable architecture for multi-database support

---

## 🛠 Tech Stack

**Frontend/UI**

* Streamlit (MVP)
* Next.js + React (Enterprise)
* Plotly (visualizations)

**Backend & Core**

* Python
* FastAPI (for APIs)
* pyodbc (MS SQL connector)

**LLM & Retrieval**

* In-house LLM / Groq API / Open-source models
* ChromaDB (schema, glossary, rules storage)
* Structured RAG for NL2SQL
* *Optional*: LlamaIndex (multi-source retrieval)
* *Optional*: LangGraph (multi-step reasoning)

**Other Services**

* APScheduler (scheduling) / Celery (enterprise jobs)
* WeasyPrint / ReportLab (PDF reports)
* SMTP/SendGrid/Slack API (report delivery)
* SQL audit tables (compliance logging)

---

## 🏗 High-Level Architecture Flow

1. User enters NL query in UI
2. Retrieve schema/glossary/rules from ChromaDB (RAG)
3. LLM generates SQL (validated for safety)
4. SQL runs on **Read-only MS SQL replica**
5. Results → Displayed in table/chart OR saved as report
6. Reports scheduled/delivered, logs stored for compliance
