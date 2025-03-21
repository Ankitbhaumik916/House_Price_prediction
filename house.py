from flask import Flask, request, render_template_string
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("model.pkl", "rb"))

# HTML Template (kept inside Python)
TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>House Price Predictor</title>
</head>
<body>
    <h2>Predict House Price</h2>
    <form action="/" method="post">
        BHK: <input type="number" name="bhk" required><br><br>
        Area (sqft): <input type="number" name="area" required><br><br>
        Distance from Metro (km): <input type="number" step="0.1" name="distance" required><br><br>
        Nearby Hospitals: <input type="number" name="hospitals" required><br><br>
        Nearby Shopping Malls: <input type="number" name="malls" required><br><br>
        Age of Property (years): <input type="number" name="age" required><br><br>
        Availability of Parking:
        <select name="parking">
            <option value="Yes">Yes</option>
            <option value="No">No</option>
        </select><br><br>
        <button type="submit">Predict</button>
    </form>

    {% if prediction %}
    <h3>Predicted Price: ₹{{ prediction }} Cr</h3>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            # Get input values
            bhk = int(request.form["bhk"])
            area = int(request.form["area"])
            distance = float(request.form["distance"])
            hospitals = int(request.form["hospitals"])
            malls = int(request.form["malls"])
            age = int(request.form["age"])
            parking = 1 if request.form["parking"] == "Yes" else 0

            # Convert input to NumPy array (reshape for model input)
            data = np.array([[bhk, area, distance, hospitals, malls, age, parking]])

            # Predict price using the trained model
            predicted_price = model.predict(data)[0]  # Assuming model returns a single value
            prediction = f"{predicted_price:.2f}"
        
        except Exception as e:
            prediction = f"Error: {e}"

    return render_template_string(TEMPLATE, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
