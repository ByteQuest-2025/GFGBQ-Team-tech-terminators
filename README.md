# GFGBQ-Team-tech-terminators
Repository for tech terminators - Vibe Coding Hackathon

## 📋 Project Summary

| Section | Details |
| :--- | :--- |
| **Problem Statement** | **PS 04:** Clinical Decision Support System (CDSS) |
| **Project Name** | AgenticDiagno |
| **Team Name** | Tech Terminators |

---

## 💡 1. Problem Statement
**PS 04 : Clinical Decision Support System**
Develop an AI-powered diagnostic support system designed to assist doctors in clinical decision-making by analyzing patient data, including medical history,reported symptoms, and laboratory results. The system leverages machine learning techniques to identify relevant disease patterns, support differential diagnosis, and highlight potential health conditions.

## 🛠️ 2. Project Name
**AgenticDiagno**

## 👥 3. Team Name
**Tech Terminators**

---

## 🔗 4. Deployed Link(Optional)
> -

## 🎬 5. 2-Minute Demonstration Video
> 📺 

## 📊 6. PPT Link
> 📂 

---

## 📖 Project Overview

The **AI Medical Symptom Analyzer** is a machine learning–powered application designed to assist users in understanding their symptoms. By leveraging **Natural Language Processing (NLP)** and **Semantic Search**, the application takes unstructured user descriptions of ailments (for example, *“I have a splitting headache and sensitivity to light”*) and matches them against a medical dataset to predict potential conditions and suggest **general medication guidelines**.

### 🔑 Key Features
- **Symptom Analysis:** Accepts natural language input describing physical symptoms.
- **AI Inference:** Uses Sentence-Transformers (`all-MiniLM-L6-v2`) for semantic similarity instead of simple keyword matching.
- **Medication Suggestions:** Provides general treatment guidance based on the predicted condition.
- **Real-Time Interface:** Built using Streamlit for a fast and interactive user experience.

---

## 🛠️ Tech Stack

- **Language:** Python 3.9+
- **Frontend:** Streamlit
- **AI Model:** Sentence-Transformers (Hugging Face)
- **Data Manipulation:** Pandas
- **Dataset:** Kaggle Symptom2Disease Dataset

---

## 📂 Project Structure

```text
medical-ai-hackathon/
├── Symptom2Disease.csv     # Medical symptom-to-disease dataset
├── fine_tune_model.ipynb   # Model experimentation & fine-tuning notebook
├── app.py                  # Streamlit application (UI layer)
├── model_logic.py          # Core ML & NLP logic (semantic search engine)
└── requirements.txt        # Project dependencies
```

---

## ⚙️ Setup & Installation

Follow the steps below to set up and run the project on your local machine.

### 1️⃣ Clone the Repository
```bash
git clone <https://github.com/ByteQuest-2025/GFGBQ-Team-tech-terminators>
cd medical-ai-hackathon
```
