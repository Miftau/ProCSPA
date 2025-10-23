from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load models and encoders
model_new = joblib.load(open("models/model_new.pkl", "rb"))
encoder_new = joblib.load(open("models/encoder_new.pkl", "rb"))

model_used = joblib.load(open("models/model_used.pkl", "rb"))
encoder_used = joblib.load(open("models/encoder_used.pkl", "rb"))

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        car_type = request.form.get("car_type")

        if car_type == "new":
            features = {
                "manufacturer": request.form.get("manufacturer"),
                "model": request.form.get("model"),
                "engine size": float(request.form.get("engine size")),
                "fuel type": request.form.get("fuel type"),
                "year of manufacture": int(request.form.get("year of manufacture")),
                "mileage": float(request.form.get("mileage")),
            }
            df = pd.DataFrame([features])
            encoded = encoder_new.transform(df)
            prediction = model_new.predict(encoded)[0]

        elif car_type == "used":
            features = {
                "brand": request.form.get("brand"),
                "model": request.form.get("model"),
                "model_year": int(request.form.get("model_year")),
                "milage": float(request.form.get("milage")),
                "fuel_type": request.form.get("fuel_type"),
                "engine": request.form.get("engine"),
                "transmission": request.form.get("transmission"),
                "ext_col": request.form.get("ext_col"),
                "int_col": request.form.get("int_col"),
                "accident": request.form.get("accident"),
                "clean_title": request.form.get("clean_title"),
                
            }
            df = pd.DataFrame([features])
            encoded = encoder_used.transform(df)
            prediction = model_used.predict(encoded)[0]
        else:
            prediction = "Invalid car type."

        return render_template("result.html", prediction=prediction)

    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
