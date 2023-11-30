from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import pandas as pd

# Loading the dataset
data_h = pd.read_csv('C:\\Users\\user\\Desktop\\kc_house_data.csv')

# Selecting the features and target variable
features = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 'zipcode']
target = 'price'
X = data_h[features]
y = data_h[target]

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model 1: Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_score = linear_model.score(X_test, y_test)

# Model 2: Random Forest Regressor
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_score = rf_model.score(X_test, y_test)

# Model 3: Support Vector Regressor
svr_model = SVR()
svr_model.fit(X_train, y_train)
svr_score = svr_model.score(X_test, y_test)

# Model 4: Gradient Boosting Regressor (Linear Boosting)
gb_model = GradientBoostingRegressor(random_state=42)
gb_model.fit(X_train, y_train)
gb_score = gb_model.score(X_test, y_test)

# Print the accuracy scores
print("Linear Regression R^2 Score:", linear_score)
print("Random Forest Regressor R^2 Score:", rf_score)
print("Support Vector Regressor R^2 Score:", svr_score)
print("Gradient Boosting Regressor R^2 Score:", gb_score)
