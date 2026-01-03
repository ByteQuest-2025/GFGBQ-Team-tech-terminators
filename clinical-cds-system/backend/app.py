from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json

app = Flask(__name__)
CORS(app)

# Load model
model = AutoModelForSequenceClassification.from_pretrained("./medical_model")
tokenizer = AutoTokenizer.from_pretrained("./medical_model")
model.eval()

# Load label mapping
with open('./label_mapping.json', 'r') as f:
    label_mapping = json.load(f)
id2label = {v: k for k, v in label_mapping.items()}


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        symptoms = data.get('symptoms', '')

        # Tokenize
        inputs = tokenizer(symptoms, return_tensors="pt",
                           truncation=True, max_length=512)

        # Predict
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)

        # Get top 3 predictions
        top3 = torch.topk(predictions, 3)
        results = []
        for i in range(3):
            disease = id2label[top3.indices[0][i].item()]
            confidence = top3.values[0][i].item() * 100
            results.append({
                "disease": disease,
                "confidence": round(confidence, 2),
                "recommended_tests": get_recommended_tests(disease)
            })

        return jsonify({
            "symptoms": symptoms,
            "predictions": results,
            "recommendations": get_general_recommendations(results[0]['disease'])
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_recommended_tests(disease):
    # Simple mapping - expand this
    test_map = {
        "Migraine": ["Neurological exam", "MRI Brain", "Blood tests"],
        "Arthritis": ["Joint X-ray", "RA factor test", "ESR test"],
        "Hypertension": ["Blood pressure monitoring", "ECG", "Renal function test"],
    }
    return test_map.get(disease, ["Complete Blood Count", "General physical examination"])


def get_general_recommendations(disease):
    return {
        "urgent": "Consult specialist within 1 week" if disease in ["Hypertension",
                                                                    "Heart Disease"] else "Routine consultation",
        "self_care": "Monitor symptoms, maintain hydration",
        "avoid": "Self-medication without doctor consultation"
    }


if __name__ == '__main__':
    app.run(debug=True, port=5000)