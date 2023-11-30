
from flask import Flask, render_template, request, jsonify
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd

app = Flask(__name__)

# Load the dataset
data_h = pd.read_csv('C:\\Users\\USER\\Desktop\\kc_house_data.csv')

# Selecting the features and target variable
Features1 = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
target = 'price'
X1 = data_h[Features1]
y1 = data_h[target]

# Data splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Linear Regression model creation
model = LinearRegression()

# Training the model
model.fit(X_train, y_train)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.json
        new_house = pd.DataFrame(data, index=[0])  # Create a DataFrame with a single row
        predicted_price = model.predict(new_house)
        return jsonify({'predicted_price': predicted_price[0]})

if __name__ == '__main__':
    app.run(debug=True)
