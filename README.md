# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load and Prepare Data
Import required libraries and load the dataset using pandas. Clean column names and separate input features (Size, Bedrooms) and target variables (Price, Occupants).

2.Scale the Features
Apply feature scaling using StandardScaler to normalize the input data, which improves the performance of SGD.

3.Initialize the Models
Create two SGD Regressor models—one for predicting price and another for predicting occupants—with suitable parameters.

4.Train the Models
Fit both models using the scaled input data and their respective target values.

5.Predict Output
Take user input (house size and bedrooms), scale it using the same scaler, and use both trained models to predict and display the price and number of occupants.

## Program:
```

import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler


data = pd.read_csv("house.csv")

data.columns = data.columns.str.strip()

X = data[['Size', 'Bedrooms']]


y_price = data['Price']
y_occ = data['Occupants']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


price_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)
occ_model = SGDRegressor(max_iter=1000, learning_rate='constant', eta0=0.01)


price_model.fit(X_scaled, y_price)
occ_model.fit(X_scaled, y_occ)


size = float(input("Enter house size: "))
bed = int(input("Enter number of bedrooms: "))


new_data = scaler.transform([[size, bed]])


pred_price = price_model.predict(new_data)
pred_occ = occ_model.predict(new_data)

print("Predicted Price:", pred_price[0])
print("Predicted Occupants:", round(pred_occ[0]))
Developed by: Rachanaa R
RegisterNumber:  212225040322
*/
```

## Output:
<img width="395" height="98" alt="Screenshot 2026-04-27 142508" src="https://github.com/user-attachments/assets/0c6a1c5e-dba9-49ce-83f1-d5c2ed625a63" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
