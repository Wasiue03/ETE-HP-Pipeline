import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

model = joblib.load('model.pkl')
model_columns = joblib.load('model_columns.pkl')  # List of columns expected

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])

    # One-hot encode or convert features as needed (same way you did during training)

    # Align input with training columns
    df = df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(df)[0]
    return jsonify({'prediction': prediction})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')