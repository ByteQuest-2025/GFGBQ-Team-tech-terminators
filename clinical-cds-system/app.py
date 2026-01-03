import streamlit as st
from model_logic import get_top_3_diagnosis, get_medicine_details, get_gemini_reasoning

st.set_page_config(page_title="MediScan AI", layout="wide")

# Initialize Session State
if 'page' not in st.session_state:
    st.session_state.page = 'profile'
if 'user_data' not in st.session_state:
    st.session_state.user_data = {}
if 'step' not in st.session_state:
    st.session_state.step = 'initial'
if 'top_3' not in st.session_state:
    st.session_state.top_3 = []
if 'initial_symptoms' not in st.session_state:
    st.session_state.initial_symptoms = ""
if 'followup_response' not in st.session_state:
    st.session_state.followup_response = ""

# --- PAGE 1: PATIENT PROFILE ---
if st.session_state.page == 'profile':
    st.title("📋 Patient Intake Form")
    st.info("Please provide your details for a personalized medical analysis.")
    
    with st.form("intake"):
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("Full Name")
            age = st.number_input("Age", 1, 100, 25)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        with c2:
            weight = st.number_input("Weight (kg)", 10.0, 200.0, 70.0)
            allergies = st.multiselect("Known Allergies", ["Penicillin", "Sulfur", "Aspirin", "None"])
            chronic = st.multiselect("Chronic Conditions", ["Diabetes", "Heart Disease", "None"])
            
        if st.form_submit_button("Continue to Diagnosis →"):
            st.session_state.user_data = {
                "name": name, "age": age, "gender": gender, 
                "weight": weight, "allergies": allergies, "chronic": chronic
            }
            st.session_state.page = 'diagnosis'
            st.session_state.step = 'initial'  # Reset to initial step
            st.rerun()

# --- PAGE 2: DIAGNOSTIC DASHBOARD ---
elif st.session_state.page == 'diagnosis':
    st.sidebar.button("← Edit Profile", on_click=lambda: st.session_state.update({"page": 'profile', "step": 'initial'}))
    
    st.title(f"🏥 Medical Analysis for {st.session_state.user_data['name']}")
    
    # Model Selector
    mode = st.radio("Model Engine:", ["Fast", "Expert"], horizontal=True)
    
    # --- STAGE 1: INITIAL SYMPTOMS ---
    if st.session_state.step == 'initial':
        st.subheader("🩺 Step 1: Describe Your Symptoms")
        user_input = st.text_area("Tell us what you're experiencing:", 
                                   value=st.session_state.initial_symptoms,
                                   height=150,
                                   placeholder="e.g., I have been experiencing fever, headache, and body aches for the past 3 days...")
        
        if st.button("🔍 Initial Analysis", type="primary"):
            if user_input.strip():
                with st.spinner("Analyzing symptoms with AI models..."):
                    # 1. Get top 3 from your local medical-trained model
                    st.session_state.top_3 = get_top_3_diagnosis(user_input, mode=mode)
                    st.session_state.initial_symptoms = user_input
                    st.session_state.step = "followup"
                    st.rerun()
            else:
                st.warning("Please describe your symptoms before proceeding.")
    
    # --- STAGE 2: FOLLOW-UP QUESTIONS ---
    elif st.session_state.step == 'followup':
        st.subheader("📊 Initial Analysis Results")
        
        # Display top 3 candidates
        col1, col2, col3 = st.columns(3)
        for idx, (col, candidate) in enumerate(zip([col1, col2, col3], st.session_state.top_3)):
            with col:
                st.metric(
                    label=f"#{idx+1} Possibility",
                    value=candidate['label'],
                    delta=f"{candidate['confidence']}% confidence"
                )
        
        st.divider()
        
        # Get Gemini's reasoning and follow-up questions
        with st.spinner("AI Doctor is analyzing your case..."):
            gemini_analysis = get_gemini_reasoning(
                st.session_state.user_data,
                st.session_state.initial_symptoms,
                st.session_state.top_3
            )
        
        st.subheader("🤖 AI Clinical Reasoning")
        st.markdown(gemini_analysis)
        
        st.divider()
        
        st.subheader("🩺 Step 2: Additional Information")
        st.info("Based on the analysis above, please answer the follow-up questions to help narrow down the diagnosis.")
        
        followup_answer = st.text_area(
            "Your answers to the follow-up questions:",
            value=st.session_state.followup_response,
            height=120,
            placeholder="Please provide detailed answers to help us make a more accurate diagnosis..."
        )
        
        col_back, col_next = st.columns([1, 1])
        with col_back:
            if st.button("← Back to Symptoms"):
                st.session_state.step = 'initial'
                st.rerun()
        
        with col_next:
            if st.button("🎯 Finalize Diagnosis", type="primary"):
                if followup_answer.strip():
                    st.session_state.followup_response = followup_answer
                    st.session_state.step = "final"
                    st.rerun()
                else:
                    st.warning("Please provide answers to the follow-up questions.")
    
    # --- STAGE 3: FINAL DIAGNOSIS & TREATMENT ---
    elif st.session_state.step == 'final':
        st.subheader("🎯 Final Diagnosis & Treatment Plan")
        
        # Get final analysis from Gemini
        with st.spinner("Generating comprehensive medical report..."):
            final_prompt = f"""
            You are a clinical decision support system providing final diagnosis.
            
            Patient Profile: Age {st.session_state.user_data['age']}, Gender {st.session_state.user_data['gender']}, 
            Chronic Conditions: {st.session_state.user_data['chronic']}, Allergies: {st.session_state.user_data['allergies']}.
            
            Initial Symptoms: {st.session_state.initial_symptoms}
            
            Top 3 AI Model Predictions:
            1. {st.session_state.top_3[0]['label']} ({st.session_state.top_3[0]['confidence']}% match)
            2. {st.session_state.top_3[1]['label']} ({st.session_state.top_3[1]['confidence']}% match)
            3. {st.session_state.top_3[2]['label']} ({st.session_state.top_3[2]['confidence']}% match)
            
            Follow-up Information Provided: {st.session_state.followup_response}
            
            Task:
            1. Provide your FINAL DIAGNOSIS with confidence level
            2. Explain why this is the most likely condition
            3. List immediate next steps
            4. Provide warning signs that require emergency care
            
            Keep the response professional and structured.
            """
            
            import google.generativeai as genai
            
            # List available models for debugging
            print("=" * 50)
            print("Available Gemini Models:")
            for m in genai.list_models():
                print(m.name)
            print("=" * 50)
            
            model = genai.GenerativeModel('gemini-1.5-flash')
            final_report = model.generate_content(final_prompt)
        
        st.markdown(final_report.text)
        
        st.divider()
        
        # Get the most likely diagnosis for medication lookup
        most_likely = st.session_state.top_3[0]['label']
        confidence = st.session_state.top_3[0]['confidence']
        
        # Display Results
        col_res, col_risk = st.columns([2, 1])
        
        with col_res:
            st.success(f"**Most Likely Condition:** {most_likely} ({confidence}% confidence)")
            
            med_info = get_medicine_details(most_likely)
            
            if med_info:
                st.subheader("💊 Medication Details")
                st.write(f"**Primary Medicine:** {med_info['medicine']}")
                st.write(f"**How it works:** {med_info['use_case']}")
                st.write(f"**Dosage Info:** {med_info['dosage']}")
                
                with st.expander("⚠️ View Warnings & Side Effects"):
                    st.warning(f"**Side Effects:** {med_info['side_effects']}")
                    st.error(f"**Important Warnings:** {med_info['warnings']}")
            else:
                st.warning("Medication data for this condition is still being updated.")

        with col_risk:
            # Risk Level Indicator
            risk = med_info['risk_level'] if med_info else "Unknown"
            st.metric("Risk Level", risk)
            
            if risk == "Critical":
                st.error("🚨 EMERGENCY: Consult a doctor immediately.")
            elif risk == "High":
                st.warning("⚠️ Serious condition: Hospital visit recommended.")
            elif risk == "Moderate":
                st.info("ℹ️ Medical attention advised: Schedule a doctor visit.")
            elif risk in ["Low", "Very Low"]:
                st.success("✅ Manageable condition: Follow dosage instructions.")
            else:
                st.info("ℹ️ Please consult a healthcare professional.")
        
        st.divider()
        
        # Action buttons
        col_restart, col_profile = st.columns([1, 1])
        with col_restart:
            if st.button("🔄 New Diagnosis"):
                st.session_state.step = 'initial'
                st.session_state.initial_symptoms = ""
                st.session_state.followup_response = ""
                st.session_state.top_3 = []
                st.rerun()
        
        with col_profile:
            if st.button("👤 Change Patient"):
                st.session_state.page = 'profile'
                st.session_state.step = 'initial'
                st.session_state.initial_symptoms = ""
                st.session_state.followup_response = ""
                st.session_state.top_3 = []
                st.rerun()

# --- FOOTER ---
st.sidebar.divider()
st.sidebar.caption("⚕️ MediScan AI - For Educational Purposes Only")
st.sidebar.caption("Always consult a licensed healthcare provider for medical advice.")