import pickle
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return "Student Performance Prediction Flask API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    input_features = [data['gender'],
                      data['race_ethnicity'],
                      data['parental_level_of_education'],
                      data['lunch'],
                      data['test_preparation_course'],
                      data['reading_score'],
                      data['writing_score']]

    prediction = model.predict([input_features])
    return jsonify({'Predicted Class': int(prediction[0])})
