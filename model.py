# create_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pickle

# Load the dataset
df = pd.read_csv('breast_cancer.csv')

# Check for the feature names and target variable
print("Feature names in the dataset:", df.columns.tolist())

# Define features and target
X = df.drop(['diagnosis', 'id'], axis=1)  # Ensure to drop 'diagnosis' and 'id'
y = df['diagnosis'].map({'M': 1, 'B': 0})  # Map 'M' to 1 and 'B' to 0

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
model = SVC()
model.fit(X_train_scaled, y_train)

# Save the model and the scaler
with open('model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)

with open('scaler.pkl', 'wb') as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved successfully.")
