TactiMerge

Overview

TactiMerge is a toolkit for deep tactical analysis and match outcome predictions in football. It leverages LLMs, vector retrieval, and a user-friendly UI to let fans and analysts explore team playstyles and project match statistics.

Features

Playstyle Analysis (/analyze): Examine team tactics across eras.

Outcome Prediction (/predict): Generate match predictions and xG estimates.

Data Ingestion: Scripts to pull and tag match reports by team and era.

Vector Retrieval: ChromaDB-backed corpus querying.

API: FastAPI endpoints for seamless integration.

UI: Gradio interface on Hugging Face Spaces.

Tech Stack

Python 3.x

LangChain

Transformers

ChromaDB

FastAPI & Uvicorn

Gradio

Prerequisites

Git

Python 3.8+

pip or venv

Installation

Clone repo:

git clone https://github.com/wicked-ctrl/TactiMerge.git
cd TactiMerge

Set up virtualenv:

python3 -m venv venv (call-out Python ≥3.8 (or whatever minimum you’ve tested))
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Usage

Data Ingestion

python data_ingestion/fetch_reports.py --league <league_or_national_team> --output data/  # e.g., premier_league, la_liga, bundesliga, england, brazil will analyze and predict --competition

Run API

uvicorn api.main:app --reload

Launch UI

gradio app.ui

Repo Structure

TactiMerge/
├── data_ingestion/
├── api/
├── ui/
├── docs/
├── tests/
├── README.md
└── requirements.txt ((e.g. fastapi==0.95.3, langchain==0.0., etc.))

Contributing

Contributions welcome! Please open issues or submit PRs.

License

MIT License

