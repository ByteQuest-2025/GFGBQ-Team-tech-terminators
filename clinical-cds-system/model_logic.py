import os
import pandas as pd
import streamlit as st
import torch
import requests
from sentence_transformers import SentenceTransformer, util
from google import genai
from dotenv import load_dotenv

# -------------------- ENV + CLIENT SETUP --------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY") 
GEOAPIFY_API_KEY = os.getenv("GEOAPIFY_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not found in .env file")

client = genai.Client(api_key=GEMINI_API_KEY)

# -------------------- DATA PREPROCESSING --------------------
@st.cache_data
def load_and_preprocess_data():
    """Load and preprocess the disease dataset"""
    df = pd.read_csv("DiseaseAndSymptoms.csv")
    
    # Clean up whitespace in headers and labels
    df.columns = df.columns.str.strip()
    df['Disease'] = df['Disease'].str.strip()
    
    # Merge all 17 symptom columns into a single natural language string
    symptom_cols = [c for c in df.columns if 'Symptom' in c]
    df['text'] = df[symptom_cols].apply(
        lambda row: ', '.join(row.dropna().astype(str).str.replace('_', ' ')), 
        axis=1
    )
    
    # Rename for consistency with application logic
    df = df.rename(columns={'Disease': 'label'})
    
    # Keep unique disease-symptom clusters for faster search
    df = df.drop_duplicates(subset=['label', 'text']).reset_index(drop=True)
    return df

df = load_and_preprocess_data()

# -------------------- AI MODELS & EMBEDDINGS --------------------
@st.cache_resource
def load_models():
    """Load sentence transformer models"""
    # Fast model for quick semantic search
    fast = SentenceTransformer("all-MiniLM-L6-v2")
    # Expert model for medical nuance
    expert = SentenceTransformer("pritamdeka/S-PubMedBert-MS-MARCO")
    return fast, expert

fast_model, expert_model = load_models()

@st.cache_resource
def load_embeddings():
    """Pre-compute embeddings for the dataset"""
    # Encode the merged symptom text for the whole dataset
    fast_emb = fast_model.encode(df["text"].tolist(), convert_to_tensor=True)
    expert_emb = expert_model.encode(df["text"].tolist(), convert_to_tensor=True)
    return fast_emb, expert_emb

fast_embeddings, expert_embeddings = load_embeddings()

# -------------------- SPECIALIST MAPPING --------------------
specialist_map = {
    "Fungal infection": "Dermatologist",
    "Allergy": "Allergist",
    "GERD": "Gastroenterologist",
    "Chronic cholestasis": "Hepatologist",
    "Drug Reaction": "Allergist",
    "Peptic ulcer diseae": "Gastroenterologist",
    "AIDS": "Infectious Disease Specialist",
    "Diabetes": "Endocrinologist",
    "Gastroenteritis": "Gastroenterologist",
    "Bronchial Asthma": "Pulmonologist",
    "Hypertension": "Cardiologist",
    "Migraine": "Neurologist",
    "Cervical spondylosis": "Orthopedic Surgeon",
    "Paralysis (brain hemorrhage)": "Neurologist",
    "Jaundice": "Hepatologist",
    "Malaria": "Infectious Disease Specialist",
    "Chicken pox": "Infectious Disease Specialist",
    "Dengue": "Infectious Disease Specialist",
    "Typhoid": "Infectious Disease Specialist",
    "hepatitis A": "Hepatologist",
    "Hepatitis B": "Hepatologist",
    "Hepatitis C": "Hepatologist",
    "Hepatitis D": "Hepatologist",
    "Hepatitis E": "Hepatologist",
    "Alcoholic hepatitis": "Hepatologist",
    "Tuberculosis": "Pulmonologist",
    "Common Cold": "General Physician",
    "Dimorphic hemmorhoids(piles)": "Proctologist",
    "Varicose veins": "Vascular Surgeon",
    "Hypothyroidism": "Endocrinologist",
    "Hyperthyroidism": "Endocrinologist",
    "Hypoglycemia": "Endocrinologist",
    "Osteoarthristis": "Orthopedic Surgeon",
    "Arthritis": "Rheumatologist",
    "(vertigo) Paroymsal Positional Vertigo": "Neurologist",
    "Acne": "Dermatologist",
    "Urinary tract infection": "Urologist",
    "Psoriasis": "Dermatologist",
    "Impetigo": "Dermatologist"
}

# -------------------- MEDICATION DATABASE --------------------
detailed_med_map = {
    "Fungal infection": {"medicine": "Clotrimazole / Ketoconazole", "use_case": "Antifungal treatment", "dosage": "Apply 2x daily", "side_effects": "Itching, redness", "risk_level": "Low", "warnings": "Keep area dry."},
    "Allergy": {"medicine": "Cetirizine / Loratadine", "use_case": "Antihistamine", "dosage": "10mg once daily", "side_effects": "Drowsiness", "risk_level": "Low", "warnings": "Avoid allergens."},
    "GERD": {"medicine": "Omeprazole / Famotidine", "use_case": "Acid reflux reduction", "dosage": "20mg before breakfast", "side_effects": "Headache", "risk_level": "Low", "warnings": "Avoid spicy food."},
    "Chronic cholestasis": {"medicine": "Ursodeoxycholic Acid", "use_case": "Bile flow improvement", "dosage": "300mg 2x daily", "side_effects": "Diarrhea", "risk_level": "Moderate", "warnings": "Regular liver tests needed."},
    "Drug Reaction": {"medicine": "Diphenhydramine / Steroids", "use_case": "Counteracting allergic reaction", "dosage": "As per severity", "side_effects": "Dizziness", "risk_level": "High", "warnings": "STOP offending drug immediately."},
    "Peptic ulcer diseae": {"medicine": "Pantoprazole + Antibiotics", "use_case": "Ulcer healing", "dosage": "40mg daily", "side_effects": "Stomach pain", "risk_level": "Moderate", "warnings": "Avoid NSAIDs (Aspirin)."},
    "AIDS": {"medicine": "Antiretroviral Therapy (ART)", "use_case": "Viral suppression", "dosage": "Lifelong daily regimen", "side_effects": "Nausea, fatigue", "risk_level": "Critical", "warnings": "Do not skip doses."},
    "Diabetes": {"medicine": "Metformin / Insulin", "use_case": "Blood sugar control", "dosage": "Varies by glucose levels", "side_effects": "Hypoglycemia", "risk_level": "High", "warnings": "Regular glucose monitoring."},
    "Gastroenteritis": {"medicine": "ORS + Loperamide", "use_case": "Rehydration", "dosage": "Frequent small sips", "side_effects": "Constipation", "risk_level": "Moderate", "warnings": "Watch for dehydration."},
    "Bronchial Asthma": {"medicine": "Salbutamol Inhaler", "use_case": "Bronchodilation", "dosage": "1-2 puffs as needed", "side_effects": "Increased heart rate", "risk_level": "Moderate", "warnings": "Keep inhaler accessible."},
    "Hypertension": {"medicine": "Amlodipine / Lisinopril", "use_case": "Blood pressure control", "dosage": "5-10mg daily", "side_effects": "Dizziness", "risk_level": "Moderate", "warnings": "Low salt diet."},
    "Migraine": {"medicine": "Sumatriptan / Naproxen", "use_case": "Pain relief", "dosage": "At onset of symptoms", "side_effects": "Tingling", "risk_level": "Low", "warnings": "Avoid light/sound triggers."},
    "Cervical spondylosis": {"medicine": "Cyclobenzaprine", "use_case": "Muscle relaxant", "dosage": "5mg 3x daily", "side_effects": "Drowsiness", "risk_level": "Low", "warnings": "Physical therapy recommended."},
    "Paralysis (brain hemorrhage)": {"medicine": "Mannitol / Anti-hypertensives", "use_case": "Reducing brain pressure", "dosage": "Hospital Setting Only", "side_effects": "Varies", "risk_level": "Critical", "warnings": "EMERGENCY: Immediate surgery/ICU."},
    "Jaundice": {"medicine": "Supportive care", "use_case": "Liver recovery", "dosage": "Hydration + Rest", "side_effects": "N/A", "risk_level": "High", "warnings": "Zero alcohol intake."},
    "Malaria": {"medicine": "Artemether + Lumefantrine", "use_case": "Anti-parasitic", "dosage": "Weight-based 3-day course", "side_effects": "Abdominal pain", "risk_level": "Critical", "warnings": "Complete the full course."},
    "Chicken pox": {"medicine": "Acyclovir", "use_case": "Anti-viral", "dosage": "800mg 5x daily", "side_effects": "Nausea", "risk_level": "Moderate", "warnings": "Highly contagious."},
    "Dengue": {"medicine": "Acetaminophen", "use_case": "Fever relief", "dosage": "500mg every 6h", "side_effects": "N/A", "risk_level": "High", "warnings": "NO Aspirin/Ibuprofen (bleeding risk)."},
    "Typhoid": {"medicine": "Azithromycin / Ceftriaxone", "use_case": "Antibiotic", "dosage": "500mg daily for 7 days", "side_effects": "Diarrhea", "risk_level": "Moderate", "warnings": "Safe water/food intake."},
    "hepatitis A": {"medicine": "Vaccination + Support", "use_case": "Viral management", "dosage": "Rest + Hydration", "side_effects": "Fatigue", "risk_level": "Moderate", "warnings": "Prevent fecal-oral spread."},
    "Hepatitis B": {"medicine": "Tenofovir", "use_case": "Viral suppression", "dosage": "300mg daily", "side_effects": "Nausea", "risk_level": "High", "warnings": "Chronic monitoring required."},
    "Hepatitis C": {"medicine": "Sofosbuvir", "use_case": "Direct Acting Antiviral", "dosage": "400mg daily", "side_effects": "Headache", "risk_level": "High", "warnings": "12-week course for cure."},
    "Hepatitis D": {"medicine": "Pegylated Interferon", "use_case": "Severe viral management", "dosage": "Injected weekly", "side_effects": "Flu-like symptoms", "risk_level": "High", "warnings": "Only occurs with Hep B."},
    "Hepatitis E": {"medicine": "Ribavirin (Severe cases)", "use_case": "Viral management", "dosage": "Supportive care", "side_effects": "N/A", "risk_level": "Moderate", "warnings": "High risk for pregnant women."},
    "Alcoholic hepatitis": {"medicine": "Corticosteroids", "use_case": "Reducing liver inflammation", "dosage": "40mg daily", "side_effects": "Weight gain", "risk_level": "High", "warnings": "ABSTAIN from alcohol."},
    "Tuberculosis": {"medicine": "Rifampicin + Isoniazid", "use_case": "Antibiotic course", "dosage": "Daily for 6-9 months", "side_effects": "Orange urine", "risk_level": "High", "warnings": "Must complete full course."},
    "Common Cold": {"medicine": "Zinc + Vitamin C", "use_case": "Immune support", "dosage": "As labeled", "side_effects": "Dry mouth", "risk_level": "Low", "warnings": "Rest and fluids."},
    "Pneumonia": {"medicine": "Amoxicillin / Levofloxacin", "use_case": "Lung infection treatment", "dosage": "500mg 3x daily", "side_effects": "Diarrhea", "risk_level": "High", "warnings": "Oxygen may be needed."},
    "Dimorphic hemmorhoids(piles)": {"medicine": "Stool Softeners + Hydrocortisone", "use_case": "Symptom relief", "dosage": "Topical + Oral", "side_effects": "Local irritation", "risk_level": "Low", "warnings": "High fiber diet."},
    "Heart attack": {"medicine": "Aspirin + Nitroglycerin", "use_case": "Blood flow restoration", "dosage": "325mg (chewable)", "side_effects": "N/A", "risk_level": "Critical", "warnings": "CALL 911/ER IMMEDIATELY."},
    "Varicose veins": {"medicine": "Sclerotherapy / Compression", "use_case": "Vein treatment", "dosage": "Wear stockings daily", "side_effects": "Skin changes", "risk_level": "Low", "warnings": "Elevate legs."},
    "Hypothyroidism": {"medicine": "Levothyroxine", "use_case": "Hormone replacement", "dosage": "Varies by blood test", "side_effects": "Palpitations", "risk_level": "Moderate", "warnings": "Take on empty stomach."},
    "Hyperthyroidism": {"medicine": "Methimazole", "use_case": "Hormone suppression", "dosage": "5-10mg daily", "side_effects": "Joint pain", "risk_level": "Moderate", "warnings": "Monitor heart rate."},
    "Hypoglycemia": {"medicine": "Glucose Tabs / Glucagon", "use_case": "Raising blood sugar", "dosage": "15g carbs immediately", "side_effects": "Confusion", "risk_level": "High", "warnings": "Follow with a meal."},
    "Osteoarthristis": {"medicine": "Acetaminophen / NSAIDs", "use_case": "Joint pain management", "dosage": "As needed", "side_effects": "Stomach upset", "risk_level": "Low", "warnings": "Gentle exercise."},
    "Arthritis": {"medicine": "Methotrexate / Ibuprofen", "use_case": "Autoimmune management", "dosage": "Weekly/Daily", "side_effects": "Nausea", "risk_level": "Moderate", "warnings": "Regular blood monitoring."},
    "(vertigo) Paroymsal Positional Vertigo": {"medicine": "Meclizine", "use_case": "Anti-vertigo", "dosage": "25mg as needed", "side_effects": "Drowsiness", "risk_level": "Low", "warnings": "Epley maneuver helps."},
    "Acne": {"medicine": "Benzoyl Peroxide / Tretinoin", "use_case": "Topical treatment", "dosage": "Apply at night", "side_effects": "Dryness", "risk_level": "Low", "warnings": "Use sunscreen."},
    "Urinary tract infection": {"medicine": "Nitrofurantoin", "use_case": "Bladder antibiotic", "dosage": "100mg 2x daily", "side_effects": "Nausea", "risk_level": "Moderate", "warnings": "Drink plenty of water."},
    "Psoriasis": {"medicine": "Clobetasol Propionate", "use_case": "Steroid cream", "dosage": "Apply thin layer", "side_effects": "Skin thinning", "risk_level": "Low", "warnings": "Avoid prolonged use."},
    "Impetigo": {"medicine": "Mupirocin Ointment", "use_case": "Bacterial skin treatment", "dosage": "3x daily", "side_effects": "Stinging", "risk_level": "Low", "warnings": "Keep sores covered."}
}

# -------------------- CORE LOGIC FUNCTIONS --------------------

def get_top_3_diagnosis(user_input: str, mode: str = "Fast"):
    """Returns top 3 unique disease predictions based on input symptoms."""
    model = fast_model if mode == "Fast" else expert_model
    embeddings = fast_embeddings if mode == "Fast" else expert_embeddings
    
    user_embedding = model.encode(user_input, convert_to_tensor=True)
    cosine_scores = util.cos_sim(user_embedding, embeddings)
    
    # Extract top 5 to filter for unique disease labels
    top_results = torch.topk(cosine_scores, k=5)
    
    unique_results = []
    seen_labels = set()
    
    for i in range(len(top_results.indices[0])):
        idx = top_results.indices[0][i].item()
        score = round(top_results.values[0][i].item() * 100, 2)
        label = df.iloc[idx]['label']
        
        if label not in seen_labels:
            unique_results.append({"label": label, "confidence": score})
            seen_labels.add(label)
        
        if len(unique_results) == 3:
            break
            
    return unique_results

def get_medicine_details(disease: str):
    """Fetches medication info from the map."""
    for key, value in detailed_med_map.items():
        if key.lower().strip() == disease.lower().strip():
            return value
    return None

def get_gemini_reasoning(user_data, user_input, candidates):
    """
    Enhanced clinical reasoning that incorporates laboratory vitals 
    to validate or flag discrepancies with BERT predictions.
    """
    # Safely extract lab results
    labs = user_data.get('labs', {})
    blood_sugar = labs.get('blood_sugar', 100)
    bp = labs.get('systolic_bp', 120)
    spo2 = labs.get('spo2', 98)
    
    # Build detailed lab summary
    lab_summary = f"""
    - Blood Glucose: {blood_sugar} mg/dL {'🚨 CRITICAL HIGH' if blood_sugar > 200 else '✓ Normal' if blood_sugar <= 140 else '⚠️ Elevated'}
    - Systolic BP: {bp} mmHg {'⚠️ HYPERTENSIVE' if bp >= 140 else '✓ Normal' if bp <= 120 else '⚠️ Elevated'}
    - Oxygen Saturation: {spo2}% {'🚨 CRITICAL LOW' if spo2 < 92 else '⚠️ Low' if spo2 < 95 else '✓ Normal'}
    """
    
    prompt = f"""
    ROLE: Senior Clinical Diagnostic Assistant performing multimodal medical analysis
    
    PATIENT PROFILE:
    - Age: {user_data.get('age')} years
    - Gender: {user_data.get('gender')}
    - Weight: {user_data.get('weight')} kg
    - Chronic Conditions: {', '.join(user_data.get('chronic', ['None']))}
    - Known Allergies: {', '.join(user_data.get('allergies', ['None']))}
    
    LABORATORY VITALS:
    {lab_summary}
    
    SYMPTOMS REPORTED: "{user_input}"
    
    AI MODEL PREDICTIONS (BERT-based semantic matching):
    {', '.join([f"{c['label']} ({c['confidence']}%)" for c in candidates])}

    CLINICAL ANALYSIS TASKS:
    
    1. **Vital-Symptom Correlation**: Cross-reference the lab vitals with both the reported symptoms AND the AI predictions.
       - Does high blood sugar (>200) align with diabetes-related predictions?
       - Does elevated BP (>140) support hypertension or related cardiovascular conditions?
       - Does low SpO2 (<95%) indicate respiratory distress that matches the symptom profile?
    
    2. **Red Flag Detection**: Identify any CRITICAL discrepancies:
       - Normal symptoms but dangerous vitals (e.g., SpO2 <92%, BP >180, Glucose >400)
       - Mismatch between predictions and vitals (e.g., AI suggests common cold but SpO2 is 88%)
       - Life-threatening combinations requiring immediate care
    
    3. **Clinical Reasoning**: Explain which of the 3 predictions is MOST LIKELY given:
       - The symptom description
       - The laboratory values
       - The patient's age and chronic history
    
    4. **Actionable Recommendations**: 
       - If vitals are critical, flag URGENT CARE NEEDED regardless of symptom severity
       - Suggest which specialist to see based on the most probable diagnosis
       - Mention any immediate monitoring needed (e.g., continuous SpO2 monitoring)
    
    FORMAT: Use clear medical language but remain empathetic. Structure your response with numbered sections.
    """
    
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        return f"⚠️ Clinical reasoning engine temporarily unavailable: {str(e)}\n\nPlease consult with a healthcare provider directly."

def get_clarifying_questions(user_input, top_candidates):
    """Gemini generates 1 critical question to distinguish between top matches."""
    prompt = f"Patient says: {user_input}. Top matches: {[c['label'] for c in top_candidates]}. Ask 1 specific medical question to distinguish between them."
    try:
        response = client.models.generate_content(model="gemini-1.5-flash", contents=prompt)
        return response.text
    except:
        return "Can you describe if the symptoms are constant or come and go?"

def get_nearby_doctors(disease_label, lat, lng):
    """Finds specialists nearby using Geoapify Places API."""
    # Get the specialist type from mapping
    specialist = specialist_map.get(disease_label, "General Physician")
    
    # Map specialists to Geoapify categories
    category = "healthcare.clinic_or_praxis"
    
    api_key = GEOAPIFY_API_KEY
    if not api_key:
        print("Warning: GEOAPIFY_API_KEY not configured")
        return []
    
    # Geoapify URL format: filter by circle (lon,lat,radius)
    url = f"https://api.geoapify.com/v2/places?categories={category}&filter=circle:{lng},{lat},5000&limit=3&apiKey={api_key}"
    
    try:
        response = requests.get(url).json()
        doctors = []
        
        # Parse Geoapify's GeoJSON structure
        for feature in response.get('features', []):
            prop = feature['properties']
            doctors.append({
                "name": prop.get('name', f"{specialist} Clinic"),
                "address": prop.get('address_line2', 'Address unavailable'),
                "rating": "N/A",
                "specialty": specialist
            })
        return doctors
    except Exception as e:
        print(f"Location error: {e}")
        return []