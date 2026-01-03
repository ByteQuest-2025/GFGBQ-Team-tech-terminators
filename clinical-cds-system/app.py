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

# ---------------- PAGE 1: PROFILE ----------------
if st.session_state.page == "profile":
    st.title("📋 Patient Intake Form")

    with st.form("intake"):
        c1, c2 = st.columns(2)

        with c1:
            name = st.text_input("Full Name")
            age = st.number_input("Age", 1, 120, 25)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])

        with c2:
            weight = st.number_input("Weight (kg)", 10.0, 300.0, 70.0)
            allergies = st.multiselect(
                "Known Allergies", ["Penicillin", "Sulfur", "Aspirin", "None"]
            )
            chronic = st.multiselect(
                "Chronic Conditions", ["Diabetes", "Heart Disease", "None"]
            )

        if st.form_submit_button("Continue →"):
            st.session_state.user_data = {
                "name": name,
                "age": age,
                "gender": gender,
                "weight": weight,
                "allergies": allergies,
                "chronic": chronic,
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
