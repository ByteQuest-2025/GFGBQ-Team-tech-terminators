import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import torch
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# List available Gemini models
def list_available_models():
    """List all available Gemini models"""
    print("Available Gemini Models:")
    for m in genai.list_models():
        print(m.name)

# Uncomment the line below to see available models when the module loads
# list_available_models()

# Load the data
df = pd.read_csv('Symptom2Disease.csv')

@st.cache_resource
def load_models():
    fast_model = SentenceTransformer('all-MiniLM-L6-v2')
    expert_model = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO') 
    return fast_model, expert_model

fast_model, expert_model = load_models()

fast_embeddings = fast_model.encode(df['text'].tolist(), convert_to_tensor=True)
expert_embeddings = expert_model.encode(df['text'].tolist(), convert_to_tensor=True)

def get_top_3_diagnosis(user_input, mode="Fast"):
    current_model = fast_model if mode == "Fast" else expert_model
    current_embeddings = fast_embeddings if mode == "Fast" else expert_embeddings
    
    user_embedding = current_model.encode(user_input, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, current_embeddings)
    
    # Get top 3 indices and scores
    top_results = torch.topk(cosine_scores, k=3)
    
    results = []
    for i in range(len(top_results.indices[0])):
        idx = top_results.indices[0][i].item()
        score = round(top_results.values[0][i].item() * 100, 2)
        label = df.iloc[idx]['label']
        results.append({"label": label, "confidence": score})
        
    return results
# --- STRUCTURED KNOWLEDGE BASE ---

def get_gemini_reasoning(user_data, user_input, candidates):
    # In model_logic.py:
    model = genai.GenerativeModel('gemini-pro')
    
    prompt = f"""
    You are a clinical reasoning assistant. 
    Patient Profile: Age {user_data['age']}, Gender {user_data['gender']}, 
    Chronic Conditions: {user_data['chronic']}, Allergies: {user_data['allergies']}.
    
    Symptoms Reported: {user_input}
    
    Our local model suggests these 3 possibilities:
    1. {candidates[0]['label']} ({candidates[0]['confidence']}% match)
    2. {candidates[1]['label']} ({candidates[1]['confidence']}% match)
    3. {candidates[2]['label']} ({candidates[2]['confidence']}% match)
    
    Task:
    1. Analyze which one is most likely given the patient's profile.
    2. Suggest 2 follow-up questions to distinguish between them.
    3. Provide a 'Doctor's Note' on potential risks.
    Keep the response structured and professional.
    """
    
    response = model.generate_content(prompt)
    return response.text

detailed_med_map = {
    "Psoriasis": {
        "medicine": "Clobetasol (Topical Steroid)",
        "use_case": "Reducing skin inflammation and scaling.",
        "dosage": "Adults: Apply thin layer 2x daily. Children: Use only mild versions (Hydrocortisone).",
        "side_effects": "Skin thinning, burning sensation.",
        "risk_level": "Low",
        "warnings": "Avoid eyes and broken skin."
    },
    "diabetes": {
        "medicine": "Metformin",
        "use_case": "Lowering blood glucose levels.",
        "dosage": "500mg-1000mg daily. Pediatric dosage requires specialist consultation.",
        "side_effects": "Nausea, diarrhea, metallic taste.",
        "risk_level": "High",
        "warnings": "Requires kidney function check before use."
    },
    "Hypertension": {
        "medicine": "Amlodipine / Lisinopril",
        "use_case": "Lowering high blood pressure.",
        "dosage": "Adults: 5mg-10mg daily. Not recommended for children without Rx.",
        "side_effects": "Dizziness, swelling in ankles.",
        "risk_level": "Moderate",
        "warnings": "Monitor heart rate regularly."
    },
    "Malaria": {
        "medicine": "Artemether + Lumefantrine",
        "use_case": "Treating parasitic infection in the blood.",
        "dosage": "Based on body weight. Complete the full 3-day course.",
        "side_effects": "Loss of appetite, muscle pain.",
        "risk_level": "Critical",
        "warnings": "Seek hospital care if vomiting persists."
    },
    "Varicose Veins": {
        "medicine": "Diosmin + Hesperidin",
        "use_case": "Improving blood circulation and reducing leg swelling.",
        "dosage": "Adults: 500mg twice daily. Not recommended for children.",
        "side_effects": "Stomach upset, headache, dizziness.",
        "risk_level": "Low",
        "warnings": "Consult doctor if pregnant or breastfeeding."
    },
    "Typhoid": {
        "medicine": "Azithromycin / Ceftriaxone",
        "use_case": "Treating bacterial infection (Salmonella typhi).",
        "dosage": "Adults: 500mg daily for 7-14 days. Children: Weight-based dosing.",
        "side_effects": "Nausea, diarrhea, abdominal pain.",
        "risk_level": "Critical",
        "warnings": "Complete full course. Seek immediate care if fever persists."
    },
    "Chicken pox": {
        "medicine": "Acyclovir",
        "use_case": "Reducing severity and duration of viral infection.",
        "dosage": "Adults: 800mg 5x daily. Children: 20mg/kg 4x daily for 5 days.",
        "side_effects": "Nausea, headache, diarrhea.",
        "risk_level": "Moderate",
        "warnings": "Start within 24 hours of rash. Avoid scratching blisters."
    },
    "Impetigo": {
        "medicine": "Mupirocin (Topical Antibiotic)",
        "use_case": "Treating bacterial skin infection.",
        "dosage": "Apply to affected area 3x daily for 5-10 days.",
        "side_effects": "Burning, stinging, itching at application site.",
        "risk_level": "Low",
        "warnings": "Keep area clean. Wash hands frequently to prevent spread."
    },
    "Dengue": {
        "medicine": "Paracetamol (Supportive Care)",
        "use_case": "Managing fever and pain. No specific antiviral treatment.",
        "dosage": "Adults: 500mg-1000mg every 4-6 hours. Children: 10-15mg/kg.",
        "side_effects": "Generally safe at recommended doses.",
        "risk_level": "Critical",
        "warnings": "AVOID aspirin and ibuprofen. Monitor platelet count. Seek hospital if bleeding occurs."
    },
    "Fungal infection": {
        "medicine": "Clotrimazole / Fluconazole",
        "use_case": "Treating fungal infections of skin, nails, or systemic.",
        "dosage": "Topical: Apply 2x daily. Oral: 150mg single dose or as prescribed.",
        "side_effects": "Skin irritation (topical), nausea (oral).",
        "risk_level": "Low",
        "warnings": "Complete full treatment course even if symptoms improve."
    },
    "Common Cold": {
        "medicine": "Paracetamol / Decongestants",
        "use_case": "Relieving symptoms like fever, congestion, body aches.",
        "dosage": "Adults: 500mg-1000mg paracetamol every 4-6 hours. Children: Age-appropriate dosing.",
        "side_effects": "Drowsiness (decongestants), stomach upset.",
        "risk_level": "Very Low",
        "warnings": "Rest and hydration are key. No antibiotics needed for viral infection."
    },
    "Pneumonia": {
        "medicine": "Amoxicillin / Azithromycin",
        "use_case": "Treating bacterial lung infection.",
        "dosage": "Adults: 500mg-1000mg every 8 hours. Children: Weight-based dosing.",
        "side_effects": "Diarrhea, nausea, allergic reactions.",
        "risk_level": "High",
        "warnings": "Seek immediate care if breathing difficulty worsens. Complete full antibiotic course."
    },
    "Dimorphic Hemorrhoids": {
        "medicine": "Hydrocortisone Cream / Fiber Supplements",
        "use_case": "Reducing inflammation and improving bowel movements.",
        "dosage": "Topical: Apply 2-3x daily. Fiber: 20-30g daily with plenty of water.",
        "side_effects": "Mild burning (topical), bloating (fiber).",
        "risk_level": "Low",
        "warnings": "If bleeding persists or severe pain, consult doctor immediately."
    },
    "Arthritis": {
        "medicine": "Ibuprofen / Naproxen (NSAIDs)",
        "use_case": "Reducing joint inflammation and pain.",
        "dosage": "Adults: 400-800mg every 6-8 hours with food. Not for children without prescription.",
        "side_effects": "Stomach upset, increased bleeding risk.",
        "risk_level": "Moderate",
        "warnings": "Long-term use requires monitoring. Avoid if history of ulcers."
    },
    "Acne": {
        "medicine": "Benzoyl Peroxide / Adapalene",
        "use_case": "Treating bacterial infection and unclogging pores.",
        "dosage": "Apply thin layer once daily at night. Start with lower concentration.",
        "side_effects": "Dryness, redness, peeling.",
        "risk_level": "Very Low",
        "warnings": "Use sunscreen during day. Avoid eyes and lips."
    },
    "Bronchial Asthma": {
        "medicine": "Salbutamol (Albuterol) Inhaler / Budesonide",
        "use_case": "Relieving airway constriction and reducing inflammation.",
        "dosage": "Rescue: 2 puffs as needed. Maintenance: As prescribed by doctor.",
        "side_effects": "Tremors, increased heart rate, throat irritation.",
        "risk_level": "Moderate",
        "warnings": "Always carry rescue inhaler. Seek emergency care if not improving."
    },
    "Migraine": {
        "medicine": "Sumatriptan / Ibuprofen",
        "use_case": "Treating acute migraine attacks and pain relief.",
        "dosage": "Sumatriptan: 50-100mg at onset. Ibuprofen: 400-600mg.",
        "side_effects": "Dizziness, drowsiness, tingling sensations.",
        "risk_level": "Moderate",
        "warnings": "Rest in dark, quiet room. Avoid triggers. Seek care if vision changes."
    },
    "Cervical spondylosis": {
        "medicine": "Ibuprofen / Cyclobenzaprine (Muscle Relaxant)",
        "use_case": "Reducing neck pain and muscle spasms.",
        "dosage": "Ibuprofen: 400-800mg 3x daily. Cyclobenzaprine: 5-10mg 3x daily.",
        "side_effects": "Drowsiness, dry mouth, dizziness.",
        "risk_level": "Low",
        "warnings": "Physical therapy recommended. Avoid prolonged static postures."
    },
    "Jaundice": {
        "medicine": "Treatment depends on underlying cause (e.g., Ursodeoxycholic acid)",
        "use_case": "Supporting liver function and reducing bilirubin levels.",
        "dosage": "As prescribed based on specific condition and severity.",
        "side_effects": "Varies based on treatment.",
        "risk_level": "High",
        "warnings": "Requires medical diagnosis. Monitor liver function tests. Avoid alcohol."
    },
    "urinary tract infection": {
        "medicine": "Nitrofurantoin / Trimethoprim",
        "use_case": "Treating bacterial infection in urinary tract.",
        "dosage": "Adults: 100mg twice daily for 5-7 days. Children: Weight-based dosing.",
        "side_effects": "Nausea, headache, brown urine discoloration.",
        "risk_level": "Moderate",
        "warnings": "Drink plenty of water. Complete full course. Seek care if fever or back pain."
    },
    "allergy": {
        "medicine": "Cetirizine / Loratadine (Antihistamines)",
        "use_case": "Relieving allergic symptoms like sneezing, itching, runny nose.",
        "dosage": "Adults: 10mg once daily. Children 6-12: 5mg once daily.",
        "side_effects": "Drowsiness (less with newer antihistamines), dry mouth.",
        "risk_level": "Very Low",
        "warnings": "Identify and avoid triggers. Carry epinephrine if severe allergies."
    },
    "gastroesophageal reflux disease": {
        "medicine": "Omeprazole / Pantoprazole (PPIs)",
        "use_case": "Reducing stomach acid production and healing esophagus.",
        "dosage": "Adults: 20-40mg once daily before breakfast. Children: Weight-based dosing.",
        "side_effects": "Headache, nausea, abdominal pain.",
        "risk_level": "Low",
        "warnings": "Avoid trigger foods. Don't lie down immediately after eating."
    },
    "drug reaction": {
        "medicine": "Diphenhydramine / Corticosteroids",
        "use_case": "Managing allergic reaction to medications.",
        "dosage": "Mild: 25-50mg diphenhydramine every 4-6 hours. Severe: Immediate medical care.",
        "side_effects": "Drowsiness, dry mouth.",
        "risk_level": "Variable (Low to Critical)",
        "warnings": "STOP the offending drug immediately. Seek emergency care if breathing difficulty or swelling."
    },
    "peptic ulcer disease": {
        "medicine": "Omeprazole + Amoxicillin + Clarithromycin (Triple Therapy)",
        "use_case": "Treating H. pylori infection and reducing stomach acid.",
        "dosage": "As prescribed, typically 7-14 day course.",
        "side_effects": "Diarrhea, nausea, metallic taste.",
        "risk_level": "Moderate",
        "warnings": "Avoid NSAIDs, alcohol, and smoking. Seek care if vomiting blood."
    }
}

def get_medicine_details(disease):
    # Case-insensitive lookup
    for key in detailed_med_map:
        if key.lower() == disease.lower():
            return detailed_med_map[key]
    return None

def get_clarifying_questions(user_input, top_candidates):
    model = genai.GenerativeModel("models/gemini-1.5-flash")
    prompt = f"""
    A user reported these symptoms: "{user_input}"
    Our model is split between: {', '.join([c['label'] for c in top_candidates])}.
    
    Ask the user ONE short, polite medical question that would help distinguish which of these is most likely. 
    Do not give a diagnosis yet. Just ask the question.
    """
    response = model.generate_content(prompt)
    return response.text