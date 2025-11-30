Intent mapping:

- "labour_overview": "unemployment", "employment", "labour market", "workforce"
- "top_occupations": "top occupations", "most common jobs"
- "households": "households", "families"
- "density": "population density", "densely populated"
- "housing_units": "housing units", "dwellings", "apartments"
- "students": "students", "school enrollment"
- "teachers": "teachers", "teaching staff"
- "higher_education": "university", "college", "higher education"

Bahrain Statistical AI Agent — System Architecture
1. Introduction

The Bahrain Statistical AI Agent is a modular and extensible data-processing and conversational AI system.
It ingests national datasets, cleans and merges them into unified master files, and answers user queries using a hybrid of:

Rule-based logic

Retrieval from master datasets

LLM fallback (ChatGPT / OpenAI API)

The system is designed to be simple for your client to maintain, safe for all datasets, and scalable for future analytics such as housemaid demand segmentation, labour market forecasting, and mobility insights.

2. High-Level Architecture

                +------------------------------+
                |   User Frontend / Chatbot    |
                | (Streamlit / Web UI / API)   |
                +--------------+---------------+
                               |
                               v
                  +------------+-------------+
                  |        NLU Router        |
                  | (Rule-based + LLM Fallback) 
                  +------------+-------------+
                               |
                               v
                +--------------+--------------+
                |        Core AI Agent        |
                | (describe_layer + querying) |
                +--------------+--------------+
                               |
        +----------------------+-----------------------+
        |                      |                       |
        v                      v                       v
+---------------+   +----------------------+   +-------------------------+
| Master Data   |   | Ingestion Pipeline   |   | External Fetching Layer |
| Repository    |   | ingest_and_prepare   |   | fetch_and_ingest        |
| (CSV files)   |   |                      |   |                         |
+-------+-------+   +----------+-----------+   +------------+------------+
        |                       |                        |
        |                       v                        |
        |             webhook_receiver (optional)         |
        +---------------- updates ------------------------+

3. Component Breakdown
3.1 Frontend / Chat UI

A simple web UI (Streamlit/HTML/JS/Flask) that:

Accepts questions

Sends queries to backend

Displays structured insights

Shows LLM answers when rule-based system has insufficient data

This part can be replaced or redesigned freely; backend is independent.

3.2 Natural Language Understanding — nlu_router.py

Responsible for:

✔ Intent Classification

Detects if question is about:

Labour market

Students / teachers

Population density

Higher education

Housing units

Occupations

Households

Or unknown → fallback to LLM

✔ Year Extraction

Automatically extracts and normalizes years from queries.

✔ LLM Fallback

If rule-based classification or agent logic fails, uses ChatGPT (GPT-4o-mini or any model) to generate:

A safe general answer

A refined version of the rule-based response

This makes the system feel like a real chatbot.

3.3 Core Agent — agent.py

Central orchestrator:

Accepts user intent & year

Calls correct describe_* function

Interacts with repository

Handles missing-data logic

Returns safe structured answers

This is where system decisions are made.

3.4 Description Layer — describe_layer.py

This file contains domain-specific explanations such as:

Labour market summaries

Students & teachers overview

Higher education institute breakdown

Households & density

Housing units

Occupation statistics

Also where future segmentation models will plug-in:

Housemaid demand segments

Area-level labour shortage clusters

Workforce prediction models

Mobility behavior insights

3.5 Repository Layer — repo.py

Acts as the system’s internal database:

Loads all master CSV files

Provides cached access

Offers easy filtering, grouping and merging

Ensures consistent schema across components

All statistical queries depend on this layer.

4. Data Ingestion Framework
4.1 Manual Ingestion

Client can put CSVs into:

data/incoming/


Then run:

python scripts/ingest_and_prepare.py --run


This will:

Detect dataset type

Map messy columns to correct fields

Normalize values

Merge cleanly into the correct master file

Prevent overwriting or corruption

4.2 Automatic Fetching — fetch_and_ingest.py

This script retrieves datasets from URLs stored in:

config/endpoints.json


Features:

Multi-attempt retry logic

File-size safety limits

Duplicate detection via checksum

Automatic import into data/incoming/

Auto-run of ingestion pipeline (optional)

Example command:
python scripts/fetch_and_ingest.py --run

Can be scheduled:

On Windows via "Task Scheduler"

Every 3 months / 6 months / annual update

Suitable for your client who has zero technical background

4.3 Ingestion Pipeline — ingest_and_prepare.py

This is the "brain" for data integration.

Responsibilities:

Identify correct master dataset

Auto-map unpredictable column names

Handle synonyms (e.g. “n”, “value”, “count”)

Normalize governorates, year, sex, nationality

Clean and dedupe rows

Append new rows safely

Prevent breaking schemas

Write stable updated master files

Safety Guarantees:

Never overwrites correct existing data

Never alters other files

Logs all actions

Dry run mode (--dry) available

4.4 Webhook Receiver — webhook_receiver.py

Optional, but powerful.

Allows automatic ingestion via HTTP:

Upload CSV (multipart)

Send URL for auto-download

Send raw CSV or base64

Trigger ingestion asynchronously

Useful if you integrate:

LMRA

Mobility providers

Third-party automatic exporters

5. Data Structure
5.1 Directory Layout

data/
  incoming/        <-- new raw files
  incoming_failed/ <-- suspicious data
  master/          <-- unified cleaned datasets
config/
  schemas.json     <-- column/field mapping rules
  endpoints.json   <-- URLs for external datasets
scripts/
  ingest_and_prepare.py
  fetch_and_ingest.py
  webhook_receiver.py
bahrain_agent/
  agent.py
  nlu_router.py
  describe_layer.py
  repo.py

6. Configuration Files
6.1 schemas.json

Defines expected structure and synonyms for fields.

Example:

“n”, “value”, “count” → numeric_fields

“gov”, “governorate”, “muharraq” → area fields

Used during ingestion to auto-detect how to map incoming CSVs.

6.2 endpoints.json

Contains list of URL sources to fetch from.

Example:

{
  "endpoints": [
    "https://data.gov.bh/…/export.csv"
  ]
}

7. Automation (every 6 months)

To fully automate refreshing datasets:

On Windows Task Scheduler:

Create a task:

Action: Start a program
Program: python
Arguments: scripts/fetch_and_ingest.py --run
Start in: C:\path\to\your\project\
Trigger: Every 6 months


This achieves:

Auto-download new data

Auto-merge into master

Auto-update model knowledge

No manual steps required.

8. Extensibility & Future Models

The architecture supports easy addition of:

✔ Housemaid Demand Segmentation

Requires:

Domestic worker permits

Household demographics

Income segments

Occupation-level salary data

Mobility clusters

Nationality distributions

✔ Labour Market Forecasting

Trend analysis using past years

Job sector growth

Occupation transitions

✔ Mobility-Driven Insights

Footfall data

Telecom mobility datasets

Cluster-based segmentation

9. Purpose of Every Important File
File	Purpose
agent.py	Main reasoning engine; connects NLU → describe layer
nlu_router.py	Intent classification + LLM fallback + answer refinement
describe_layer.py	All statistic/summary logic lives here
repo.py	Loading & querying master datasets
ingest_and_prepare.py	Clean + normalize + merge incoming data
fetch_and_ingest.py	Auto-download from endpoints.json
webhook_receiver.py	API for external ingestion triggers
schemas.json	Field mapping and schema rules
endpoints.json	URLs for automatic fetching
data/master/	Final unified datasets
architecture.md	Documentation for developers/clients