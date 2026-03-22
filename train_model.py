import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import joblib
import os

# 1. Setup Directories (Makes it a "Bigger Project")
os.makedirs('artifacts', exist_ok=True)

# 2. Load Dataset from the new 'data' folder
try:
    df = pd.read_csv('data/USA_Housing.csv')
except FileNotFoundError:
    print("🚨 Error: Please create a 'data' folder and put 'USA_Housing.csv' inside it.")
    exit()

# 3. Define Exact Features 
features = [
    'Avg. Area Income', 
    'Avg. Area House Age', 
    'Avg. Area Number of Rooms', 
    'Avg. Area Number of Bedrooms', 
    'Area Population'
]
target = 'Price'

X = df[features]
y = df[target]

# 4. Handle Missing Values
imputer = SimpleImputer(strategy='mean')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)

# 5. Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# 6. Apply Linear Regression
model = LinearRegression()
print("Training the Linear Regression model...")
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Model R^2 Score: {score:.4f}")

# 7. Save Assets to the 'artifacts' folder
joblib.dump(model, 'artifacts/house_model.pkl')
joblib.dump(imputer, 'artifacts/imputer.pkl')
joblib.dump(features, 'artifacts/features.pkl')
print("✅ Model artifacts saved successfully in the /artifacts directory!")