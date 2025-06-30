@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Load model here instead of globally
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    input_features = [data['gender'], data['race_ethnicity'], data['parental_level_of_education'],
                      data['lunch'], data['test_preparation_course'],
                      data['reading_score'], data['writing_score']]

    prediction = model.predict([input_features])
    return jsonify({'Predicted Class': int(prediction[0])})
