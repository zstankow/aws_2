from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
churn_model = pickle.load(open('churn_model.pkl', 'rb'))


@app.route('/predict_churn', methods=['GET'])
def predict_api():
    is_male = request.args.get('is_male', default=0, type=int)
    num_inters = request.args.get('num_inters', default=0, type=int)
    late_on_payment = request.args.get('late_on_payment', default=0, type=int)
    age = request.args.get('age', default=30, type=float)
    years_in_contract = request.args.get('years_in_contract', default=0, type=float)

    features = [is_male, num_inters, late_on_payment, age, years_in_contract]
    prediction = churn_model.predict(np.array([features]))
    output = prediction[0]
    return str(output)


@app.route('/predict_churn_bulk', methods=['POST'])
def predict_api_bulk():
    data = request.get_json()
    formatted_data = [np.array([d['is_male'], d['num_inters'], d['late_on_payment'], d['age'], d['years_in_contract']])
                      for d in data]
    prediction = churn_model.predict(formatted_data)
    output = prediction.tolist()
    return jsonify(output)


if __name__ == '__main__':
    app.run(debug=True, port=8080, host='0.0.0.0')
