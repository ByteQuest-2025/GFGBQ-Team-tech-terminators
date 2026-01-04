import streamlit as st
from model_logic import (
    get_top_3_diagnosis,
    get_medicine_details,
    get_gemini_reasoning,
)

st.set_page_config(page_title="MediScan AI", layout="wide")

# ---------------- SESSION STATE ----------------
if "page" not in st.session_state:
    st.session_state.page = "profile"
if "user_data" not in st.session_state:
    st.session_state.user_data = {}
if "step" not in st.session_state:
    st.session_state.step = "initial"
if "top_3" not in st.session_state:
    st.session_state.top_3 = []
if "initial_symptoms" not in st.session_state:
    st.session_state.initial_symptoms = ""
if "followup_response" not in st.session_state:
    st.session_state.followup_response = ""

# ---------------- PAGE 1: PROFILE (UPDATED WITH LAB VITALS) ----------------
if st.session_state.page == "profile":
    st.title("📋 Patient Intake Form")
    st.caption("Complete patient information for accurate clinical assessment")

    with st.form("intake"):
        # Use 3 columns to organize inputs including lab results
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**👤 Basic Information**")
            name = st.text_input("Full Name", placeholder="John Doe")
            age = st.number_input("Age", 1, 120, 25)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])

        with c2:
            st.markdown("**📊 Medical History**")
            weight = st.number_input("Weight (kg)", 10.0, 300.0, 70.0)
            allergies = st.multiselect(
                "Known Allergies", 
                ["Penicillin", "Sulfur", "Aspirin", "Latex", "None"]
            )
            chronic = st.multiselect(
                "Chronic Conditions", 
                ["Diabetes", "Hypertension", "Heart Disease", "Asthma", "None"]
            )

        with c3:
            st.markdown("**🔬 Laboratory Vitals** *(Optional)*")
            st.caption("Leave at default if unavailable")
            
            blood_sugar = st.number_input(
                "Blood Glucose (mg/dL)", 
                min_value=0, 
                max_value=600, 
                value=100,
                help="Normal: 70-100 (fasting), <140 (post-meal)"
            )
            
            systolic_bp = st.number_input(
                "Systolic BP (mmHg)", 
                min_value=0, 
                max_value=250, 
                value=120,
                help="Normal: 90-120 mmHg"
            )
            
            oxygen_level = st.number_input(
                "SpO2 - Oxygen Saturation (%)", 
                min_value=0, 
                max_value=100, 
                value=98,
                help="Normal: 95-100%"
            )

        if st.form_submit_button("Continue to Diagnosis →", type="primary"):
            # Validate critical inputs
            if not name.strip():
                st.error("❌ Patient name is required")
                st.stop()
            
            # Store all data including labs
            st.session_state.user_data = {
                "name": name,
                "age": age,
                "gender": gender,
                "weight": weight,
                "allergies": allergies,
                "chronic": chronic,
                "labs": {
                    "blood_sugar": blood_sugar,
                    "systolic_bp": systolic_bp,
                    "spo2": oxygen_level
                }
            }
            st.session_state.page = "diagnosis"
            st.session_state.step = "initial"
            st.rerun()

# ---------------- PAGE 2: DIAGNOSIS ----------------
elif st.session_state.page == "diagnosis":
    st.sidebar.button(
        "← Edit Profile",
        on_click=lambda: st.session_state.update(
            {"page": "profile", "step": "initial"}
        ),
    )

    st.title(f"🏥 Medical Analysis for {st.session_state.user_data['name']}")
    
    # Display Vitals Dashboard
    if st.session_state.get('user_data', {}).get('labs'):
        st.subheader("🔬 Patient Vital Signs")
        
        labs = st.session_state.user_data['labs']
        col1, col2, col3 = st.columns(3)
        
        # Blood Sugar Metric
        with col1:
            sugar_status = "🚨 Critical" if labs['blood_sugar'] > 200 else "⚠️ High" if labs['blood_sugar'] > 140 else "✅ Normal"
            st.metric(
                "Blood Glucose", 
                f"{labs['blood_sugar']} mg/dL",
                delta=sugar_status,
                delta_color="inverse" if labs['blood_sugar'] > 140 else "normal"
            )
        
        # Blood Pressure Metric
        with col2:
            bp_status = "🚨 Critical" if labs['systolic_bp'] >= 180 else "⚠️ High" if labs['systolic_bp'] > 130 else "✅ Normal"
            st.metric(
                "Systolic BP", 
                f"{labs['systolic_bp']} mmHg",
                delta=bp_status,
                delta_color="inverse" if labs['systolic_bp'] > 130 else "normal"
            )
        
        # Oxygen Saturation Metric
        with col3:
            spo2_status = "🚨 Critical" if labs['spo2'] < 92 else "⚠️ Low" if labs['spo2'] < 95 else "✅ Normal"
            st.metric(
                "Oxygen Saturation", 
                f"{labs['spo2']}%",
                delta=spo2_status,
                delta_color="inverse" if labs['spo2'] < 95 else "normal"
            )
        
        st.divider()
    
    mode = st.radio("Model Engine", ["Fast", "Expert"], horizontal=True)

    # ---------- STEP 1 ----------
    if st.session_state.step == "initial":
        st.subheader("🩺 Describe Your Symptoms")

        user_input = st.text_area(
            "Symptoms",
            value=st.session_state.initial_symptoms,
            height=150,
        )

        if st.button("Analyze", type="primary"):
            if not user_input.strip():
                st.warning("Symptoms cannot be empty.")
                st.stop()

            with st.spinner("Running medical models..."):
                st.session_state.top_3 = get_top_3_diagnosis(user_input, mode)
                st.session_state.initial_symptoms = user_input
                st.session_state.step = "followup"
                st.rerun()

    # ---------- STEP 2 ----------
    elif st.session_state.step == "followup":
        st.subheader("📊 Initial Results")

        cols = st.columns(3)
        for i, c in enumerate(st.session_state.top_3):
            cols[i].metric(
                f"Option {i+1}", c["label"], f"{c['confidence']}%"
            )

        with st.spinner("AI clinician reasoning..."):
            try:
                gemini_text = get_gemini_reasoning(
                    st.session_state.user_data,
                    st.session_state.initial_symptoms,
                    st.session_state.top_3,
                )
            except Exception as e:
                st.error(str(e))
                st.stop()

        st.markdown(gemini_text)

        st.subheader("Follow-up Details")
        followup = st.text_area(
            "Your response",
            value=st.session_state.followup_response,
            height=120,
        )

        if st.button("Finalize Diagnosis", type="primary"):
            if not followup.strip():
                st.warning("Follow-up info required.")
                st.stop()

            st.session_state.followup_response = followup
            st.session_state.step = "final"
            st.rerun()

    # ---------- STEP 3 ----------
    elif st.session_state.step == "final":
        st.subheader("🎯 Final Assessment")

        most_likely = st.session_state.top_3[0]["label"]
        confidence = st.session_state.top_3[0]["confidence"]

        st.success(
            f"Most Likely Condition: **{most_likely}** ({confidence}%)"
        )

        med = get_medicine_details(most_likely)
        if med:
            st.subheader("💊 Treatment Info")
            st.write("**Medicine:**", med["medicine"])
            st.write("**Use:**", med["use_case"])
            st.write("**Dosage:**", med["dosage"])

            with st.expander("Warnings"):
                st.warning(med["side_effects"])
                st.error(med["warnings"])

            st.metric("Risk Level", med["risk_level"])
        else:
            st.warning("Medication data unavailable.")

        if st.button("🔄 Start Over"):
            st.session_state.clear()
            st.rerun()

# ---------------- FOOTER ----------------
st.sidebar.divider()
st.sidebar.caption("⚠️ Educational use only. Not medical advice.")
st.sidebar.warning("""
    **⚠️ MEDICAL DISCLAIMER**
    
    This system is for educational and clinical decision SUPPORT only. 
    
    🚨 **Critical Vitals Require Immediate Action:**
    - SpO2 < 90%: Call emergency services
    - BP > 180 mmHg: Seek urgent care
    - Blood Sugar > 400 mg/dL: Emergency evaluation needed
    
    Always consult licensed healthcare providers for medical decisions.
""")