from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
encoder = pickle.load(open("encoder.pkl", "rb"))

@app.route("/")
def home():
    return "House Price Prediction API Running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    
    area = data["area"]
    bedrooms = data["bedrooms"]
    bathrooms = data["bathrooms"]
    location = encoder.transform([data["location"]])[0]
    
    prediction = model.predict([[area, bedrooms, bathrooms, location]])
    
    return jsonify({"Predicted Price": prediction[0]})

if __name__ == "__main__":
    app.run(debug=True)