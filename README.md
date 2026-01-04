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

# 🩺 AgenticDiagno (MediScan AI)
## An Agentic Clinical Decision Support System (CDSS) for the Vibe Coding Hackathon

| Section | Details |
|---------|---------|
| **Team Name** | Tech Terminators |
| **Problem Statement** | PS 04: AI-powered Clinical Decision Support System |
| **Core Tech** | Python, Streamlit, PubMedBert, Google Gemini 1.5 Flash |
| **Location Services** | Geoapify Places API |

## 💡 1. Problem Statement

Modern healthcare faces a challenge in quickly synthesizing patient symptoms into accurate differential diagnoses. AgenticDiagno is designed as a co-pilot for clinicians. It analyzes unstructured patient descriptions, correlates them with medical history, and provides evidence-based clinical analysis to reduce diagnostic errors and suggest appropriate specialists.

## 🛠️ 2. Key Features

### 🧠 Hybrid Dual-Model AI
We utilize a two-stage semantic search approach to ensure both speed and clinical accuracy:
- **Fast Mode**: Uses `all-MiniLM-L6-v2` for rapid symptom mapping.
- **Expert Mode**: Leverages `S-PubMedBert-MS-MARCO`, a model fine-tuned on medical literature, to understand complex clinical nuances and professional terminology.

### 🏥 Expanded Disease Coverage
The system has been upgraded to handle **41 unique diseases**, ranging from common ailments like the Common Cold to critical conditions like Paralysis (brain hemorrhage) and AIDS.

### 🤖 Agentic Reasoning & Follow-up
- **Clinical Reasoning**: Powered by Google Gemini 1.5 Flash, the system provides a narrative "Clinician Reasoning" block that explains why a specific diagnosis was suggested based on the patient's age and history.
- **Dynamic Questioning**: The AI generates specific follow-up questions to distinguish between closely related conditions (e.g., differentiating between Malaria and Dengue).

### 📍 Specialist Locator (Geoapify)
Integrated location-based services to find the nearest specialists (Dermatologists, Cardiologists, etc.) based on the predicted condition and the user's real-time GPS coordinates.

### 📜 Professional Reporting
- **PDF Summary**: Generates a clinical-grade PDF report including predicted conditions, medication guidelines, risk levels, and contraindication alerts.
- **Contraindication Engine**: Automatically flags medications that conflict with the patient's known allergies entered during intake.

## 📂 Project Structure

```text
medical-ai-hackathon/
├── DiseaseAndSymptoms.csv # 41-disease dataset with 17 symptom vectors
├── model_logic.py # Core BERT embeddings, medication maps, & Gemini Logic
├── app.py # Streamlit UI Layer (Multi-step intake)
├── index.html # Advanced Frontend with Body Mapping & Voice Dictation
└── requirements.txt # Torch, Transformers, GenAI, Streamlit
```

## ⚙️ Technical Workflow

1. **Patient Intake**: User enters age, gender, weight, and chronic history (e.g., Diabetes, Hypertension).
2. **Symptom Mapping**: User uses the Interactive Body Map or Voice Dictation to describe their condition.
3. **Semantic Inference**: The system calculates cosine similarity between the input and the `DiseaseAndSymptoms.csv` vector space using BERT.
4. **Risk Assessment**: Conditions are categorized by risk levels (Low to Critical).
5. **Finalization**: Gemini validates the findings, asks follow-up questions, and the system recommends medication and local doctors.

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ByteQuest-2025/GFGBQ-Team-tech-terminators
cd medical-ai-hackathon
