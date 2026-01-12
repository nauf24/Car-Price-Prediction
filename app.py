from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load("car_price_model.pkl")

# Load CSV once at startup
car_data = pd.read_csv("used_cars_cleaned.csv")

# Print the columns in the CSV for debugging purposes
print("CSV Columns:", car_data.columns.tolist())

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/how-it-works')
def how_it_works():
    return render_template('how-it-works.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    try:
        # Get unique values for dropdowns
        brands = sorted(car_data['brand'].dropna().unique())
        fuel_types = sorted(car_data['fuel_type'].dropna().unique())
        transmissions = sorted(car_data['transmission_type'].dropna().unique())

        # Print unique values to the console for debugging
        print("Unique Brands:", brands)
        print("Unique Fuel Types:", fuel_types)
        print("Unique Transmissions:", transmissions)
        
    except Exception as e:
        return f"Error loading dropdowns: {e}"

    if request.method == 'POST':
        try:
            data = pd.DataFrame([{
                'brand': request.form['brand'],
                'fuel_type': request.form['fuel_type'],
                'transmission_type': request.form['transmission_type'],
                'kms_driven': int(request.form['kms_driven']),
                'year_of_registration': int(request.form['year_of_registration']),
                'previous_owners': int(request.form['previous_owners']),
                'ex_showroom_price': float(request.form['ex_showroom_price'])
            }])
            prediction = model.predict(data)[0]
            return render_template('result.html', price=round(prediction, 2))
        except Exception as e:
            return f"Prediction error: {e}"

    return render_template('predict.html', brands=brands, fuel_types=fuel_types, transmissions=transmissions)


if __name__ == '__main__':
    app.run(debug=True)
