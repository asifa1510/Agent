# Stock Trading Agent with Sentiment-Driven Predictions

**AI trading agent merging sentiment, news, and data for explainable decisions.**

---

## 🚀 Overview
This project builds an **AI-powered stock trading agent** that integrates **social media sentiment**, **financial news**, and **historical market data** to make **autonomous, explainable trading decisions**.  
Developed for the **AWS Global Vibe: AI Coding Hackathon 2025**, it targets the **Fintech** and **Agentic AI Systems** tracks.

---

## 🧩 Problem
Retail investors face:
- **Market volatility** and unpredictable swings  
- **Biased or sensationalized news**  
- **Lack of integrated real-time insights**

This agent tackles these issues through **data-driven predictions** and **transparent decision explanations**.

---

## ✨ Features
- **Sentiment Analysis** — Real-time NLP on X/Twitter to gauge market mood  
- **News Integration** — Financial news relevance scoring via NewsAPI or RSS  
- **Historical Analysis** — LSTM/XGBoost forecasting on price trends  
- **Autonomous Trading** — Simulated trade execution with risk controls  
- **Scenario Simulation** — Monte Carlo portfolio testing  
- **Explainable Insights** — Natural-language rationales from AWS Bedrock  
- **AWS-Native Stack** — SageMaker · Lambda · Kinesis · S3 for scalability  

---

## 🏗️ Architecture
**Data Ingestion**
- Social media (X/Twitter API)  
- Financial news feeds / NewsAPI  
- Historical data (Yahoo Finance API)  
- Streamed through **Kinesis**, stored in **S3**

**Processing**
- **BERT** for sentiment  
- **LSTM/XGBoost** for time-series prediction  
- Combined signals via **FastAPI backend**  
- **Bedrock** generates trade explanations

**Frontend**
- **React + Tailwind CSS** dashboard showing predictions, trades, and insights

---

## 🧰 Tech Stack
| Layer | Technologies |
|-------|---------------|
| Frontend | React · Tailwind CSS · Chart.js |
| Backend | Python · FastAPI |
| ML | AWS SageMaker (BERT, LSTM) |
| Data Pipelines | AWS Kinesis · Lambda |
| Storage | AWS S3 · DynamoDB |
| Explainability | AWS Bedrock |
| Data Sources | Yahoo Finance · NewsAPI · X/Twitter API |

---

## ⚙️ Setup
```bash
git clone https://github.com/your-repo/stock-trading-agent.git
cd stock-trading-agent

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
