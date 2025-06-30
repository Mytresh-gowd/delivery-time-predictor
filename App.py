from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model and scaler
model = pickle.load(open("C:\\Users\\mytre\\OneDrive\\Desktop\\Project - ML 2025\\venv\\rf.pkl", "rb"))
scaler = pickle.load(open("C:\\Users\\mytre\\OneDrive\\Desktop\\Project - ML 2025\\venv\\ss.pkl", "rb"))

# Function to safely encode categorical values
def safe_label_encode(le, val):
    if val in le.classes_:
        return le.transform([val])[0]
    else:
        # Use a fallback to the first class (most common)
        return le.transform([le.classes_[0]])[0]

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from form
        age = float(request.form['age'])
        ratings = float(request.form['ratings'])
        rest_lat = float(request.form['rest_lat'])
        rest_lon = float(request.form['rest_lon'])
        del_lat = float(request.form['del_lat'])
        del_lon = float(request.form['del_lon'])
        time_ordered = float(request.form['time_ordered'])
        time_picked = float(request.form['time_picked'])
        weather = request.form['weather']
        traffic = request.form['traffic']
        vehicle_condition = float(request.form['vehicle_condition'])
        order_type = request.form['order_type']
        vehicle_type = request.form['vehicle_type']
        multiple_deliveries = float(request.form['multiple_deliveries'])
        festival = request.form['festival']
        city = request.form['city']
        distance = float(request.form['distance'])

        # Load label encoders
        le_weather = pickle.load(open("C:\\Users\\mytre\\OneDrive\\Desktop\\Project - ML 2025\\venv\\Weatherconditions.pkl", "rb"))
        le_traffic = pickle.load(open("C:\\Users\\mytre\\OneDrive\\Desktop\\Project - ML 2025\\venv\\Road_traffic_density.pkl", "rb"))
        le_order = pickle.load(open("C:\\Users\\mytre\\OneDrive\\Desktop\\Project - ML 2025\\venv\\Type_of_order.pkl", "rb"))
        le_vehicle = pickle.load(open("C:\\Users\\mytre\\OneDrive\\Desktop\\Project - ML 2025\\venv\\Type_of_vehicle.pkl", "rb"))  # fixed filename
        le_festival = pickle.load(open("C:\\Users\\mytre\\OneDrive\\Desktop\\Project - ML 2025\\venv\\Festival.pkl", "rb"))
        le_city = pickle.load(open("C:\\Users\\mytre\\OneDrive\\Desktop\\Project - ML 2025\\venv\\City.pkl", "rb"))

        # Encode categorical features
        weather_enc = safe_label_encode(le_weather, weather)
        traffic_enc = safe_label_encode(le_traffic, traffic)
        order_enc = safe_label_encode(le_order, order_type)
        vehicle_enc = safe_label_encode(le_vehicle, vehicle_type)
        festival_enc = safe_label_encode(le_festival, festival)
        city_enc = safe_label_encode(le_city, city)

        # Prepare final input
        final_input = np.array([[age, ratings, rest_lat, rest_lon,
                                 del_lat, del_lon, time_ordered, time_picked,
                                 weather_enc, traffic_enc, vehicle_condition,
                                 order_enc, vehicle_enc, multiple_deliveries,
                                 festival_enc, city_enc, distance]])

        # Scale and predict
        final_input_scaled = scaler.transform(final_input)
        prediction = model.predict(final_input_scaled)[0]

        return render_template("index.html", prediction_text=f"Predicted Time: {round(prediction, 2)} minutes")

    except Exception as e:
        return render_template("index.html", prediction_text=f"Error occurred: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
