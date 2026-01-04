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

# 🩺 AgenticDiagno (MediScan AI)
### An Agentic Clinical Decision Support System (CDSS)

> **Repository:** GFGBQ-Team-tech-terminators  
> **Hackathon:** Vibe Coding Hackathon  
> **Problem Statement:** PS 04 – Clinical Decision Support System (CDSS)

---

## 📋 Project Summary

| Section | Details |
|------|---------|
| **Project Name** | AgenticDiagno (MediScan AI) |
| **Team Name** | Tech Terminators |
| **Problem Statement** | PS 04 – AI-powered Clinical Decision Support System |
| **Core Technologies** | Python, Streamlit, PubMedBERT, Google Gemini 1.5 Flash |
| **Location Services** | Geoapify Places API |
| **Domain** | Healthcare AI / Clinical Decision Support |

---

## 👥 Team Details

**Team Name:** Tech Terminators  

> A multidisciplinary team focused on building reliable, ethical, and clinically relevant AI systems for healthcare decision support.

---

## 💡 Problem Statement (PS 04)

Modern healthcare systems face significant challenges in **accurate and timely diagnosis**, especially when clinicians must analyze:
- Unstructured patient symptom descriptions  
- Past medical history  
- Laboratory findings  
- Risk factors and contraindications  

Due to time constraints and information overload, diagnostic errors and delayed referrals remain a concern.

**AgenticDiagno** aims to solve this problem by acting as an **AI-powered clinical co-pilot** that assists healthcare professionals in:
- Synthesizing patient data  
- Generating evidence-based differential diagnoses  
- Identifying potential risks  
- Suggesting appropriate specialists  

⚠️ *Note: This system is designed to assist clinicians and does not replace professional medical judgment.*

---

## 🧠 Project Overview

### 🩺 What is AgenticDiagno?

**AgenticDiagno (MediScan AI)** is an **Agentic Clinical Decision Support System** that analyzes patient symptoms, medical history, and contextual information using **hybrid AI reasoning**.

It combines:
- **Fast semantic symptom matching**
- **Medical literature–aware reasoning**
- **Agentic follow-up questioning**
- **Location-based specialist recommendations**

The system provides **transparent clinical reasoning**, improving trust and interpretability.

---

## ✨ Key Features

### 🧠 Hybrid Dual-Model AI Architecture

AgenticDiagno uses a **two-stage semantic inference pipeline**:

#### ⚡ Fast Mode (Speed-Oriented)
- Model: `all-MiniLM-L6-v2`
- Purpose:
  - Rapid symptom embedding
  - Initial disease candidate generation
- Benefit:
  - Low latency
  - Real-time interaction

#### 🧪 Expert Mode (Clinical Accuracy)
- Model: `S-PubMedBert-MS-MARCO`
- Purpose:
  - Deep medical context understanding
  - Accurate clinical nuance interpretation
- Benefit:
  - Medical literature–aligned reasoning

---

### 🏥 Expanded Disease Coverage

- Supports **41 clinically relevant diseases**
- Covers:
  - Common conditions (Common Cold, Migraine, Gastritis)
  - Infectious diseases (Malaria, Dengue, AIDS)
  - Neurological and critical cases (Brain Hemorrhage, Paralysis)
- Each disease is mapped with **17 symptom vectors**

---

### 🤖 Agentic Reasoning & Intelligent Follow-ups

Powered by **Google Gemini 1.5 Flash**, the system demonstrates **agentic behavior**:

- 🧠 **Clinician Reasoning Block**
  - Explains *why* a diagnosis was suggested
  - Considers age, gender, medical history, and symptom severity

- ❓ **Dynamic Follow-up Questions**
  - Automatically generated to disambiguate similar diseases  
  - Example:
    - Differentiates *Malaria vs Dengue* based on fever pattern, platelet indicators, and body pain

---

### 📍 Specialist Locator (Geoapify Integration)

- Uses **Geoapify Places API**
- Detects real-time location
- Recommends nearest:
  - Cardiologists
  - Neurologists
  - Dermatologists
  - General Physicians
- Enables faster clinical referrals

---

### 📜 Professional Clinical Reporting

- 🧾 **PDF Medical Summary**
  - Predicted conditions
  - Risk classification (Low → Critical)
  - Medication guidance
  - Contraindication alerts

- 🚫 **Contraindication Engine**
  - Cross-checks patient allergies
  - Flags unsafe medications automatically

---

## ⚙️ Technical Workflow

1. **Patient Intake**
   - Age, gender, weight
   - Chronic conditions (Diabetes, Hypertension, etc.)
   - Known allergies

2. **Symptom Collection**
   - Free-text description
   - Interactive body map
   - Voice dictation

3. **Semantic Inference**
   - BERT embeddings
   - Cosine similarity against disease vectors

4. **Risk Stratification**
   - Conditions categorized from *Low* to *Critical*

5. **Agentic Validation**
   - Gemini validates results
   - Generates reasoning + follow-up questions

6. **Final Output**
   - Medication guidance
   - Specialist recommendations
   - Downloadable clinical PDF report

---

## 📂 Project Structure

```text
medical-ai-hackathon/
├── DiseaseAndSymptoms.csv      # 41-disease dataset with symptom vectors
├── model_logic.py              # BERT embeddings, medication logic, Gemini reasoning
├── app.py                      # Streamlit multi-step clinical UI
├── index.html                  # Advanced frontend (body map & voice input)
├── requirements.txt            # Dependencies (Torch, Transformers, Streamlit, GenAI)
└── README.md                   # Project documentation
```

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/ByteQuest-2025/GFGBQ-Team-tech-terminators
cd medical-ai-hackathon
