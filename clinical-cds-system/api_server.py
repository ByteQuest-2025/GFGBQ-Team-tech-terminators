from flask import Flask, request, jsonify
from flask_cors import CORS

# Lazy-import model logic to avoid heavy model download at startup
app = Flask(__name__)
CORS(app)  # Enable CORS for HTML frontend

@app.route("/", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "service": "Diagnosense API", "version": "2.0"})

@app.route('/api/diagnosis', methods=['POST'])
def api_get_diagnosis():
    """Get top 3 diagnosis predictions"""
    data = request.json or {}
    user_input = data.get('symptoms', '')
    mode = data.get('mode', 'Fast')
    selected_body = data.get('selectedBodyPart')

    if not user_input.strip():
        return jsonify({'error': 'Symptoms are required'}), 400

    try:
        # Import here to defer heavy model loading until first request
        from model_logic import get_top_3_diagnosis
        diagnosis = get_top_3_diagnosis(user_input, mode)
        return jsonify({
            'diagnosis': diagnosis, 
            'mode': mode,
            'selectedBodyPart': selected_body
        })
    except Exception as e:
        print(f"Diagnosis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reasoning', methods=['POST'])
def api_get_reasoning():
    """Get AI clinical reasoning based on user data and candidates"""
    data = request.json or {}
    user_data = data.get('user_data', {})
    user_input = data.get('symptoms', '')
    candidates = data.get('candidates', [])

    if not user_input or not candidates:
        return jsonify({'error': 'Symptoms and candidates are required'}), 400

    try:
        # Import here to defer heavy model loading until first request
        from model_logic import get_gemini_reasoning
        reasoning = get_gemini_reasoning(user_data, user_input, candidates)
        return jsonify({'reasoning': reasoning})
    except Exception as e:
        print(f"Reasoning error: {str(e)}")
        return jsonify({'error': str(e), 'reasoning': f"Clinical analysis unavailable: {str(e)}"}), 500

@app.route('/api/medicine/<disease>', methods=['GET'])
def api_get_medicine(disease):
    """Get medication details for a specific disease"""
    try:
        from model_logic import get_medicine_details
        med_info = get_medicine_details(disease)
        
        if med_info:
            return jsonify({'disease': disease, 'medicine': med_info})
        else:
            return jsonify({'error': 'Medicine information not found'}), 404
    except Exception as e:
        print(f"Medicine lookup error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/doctors', methods=['POST'])
def api_get_doctors():
    """Get nearby doctors based on disease and location"""
    data = request.json or {}
    disease = data.get('disease', '')
    lat = data.get('lat')
    lng = data.get('lng')

    if not disease or lat is None or lng is None:
        return jsonify({'error': 'Disease and location (lat, lng) are required'}), 400

    try:
        from model_logic import get_nearby_doctors
        doctors = get_nearby_doctors(disease, lat, lng)
        return jsonify({'doctors': doctors, 'disease': disease})
    except Exception as e:
        print(f"Doctor lookup error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("🚀 Starting Diagnosense API Server...")
    print("📡 API Endpoints:")
    print("   - GET  /              : Health check")
    print("   - POST /api/diagnosis : Get diagnosis predictions")
    print("   - POST /api/reasoning : Get AI clinical reasoning")
    print("   - GET  /api/medicine/<disease> : Get medicine details")
    print("   - POST /api/doctors   : Get nearby doctors")
    print("\n🌐 Server running on http://localhost:5000")
    
    # Use 0.0.0.0 for container friendliness
    app.run(host='0.0.0.0', port=5000, debug=True)