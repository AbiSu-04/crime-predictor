import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Load data
df = pd.read_csv("crime-data-from-2010-to-present.csv", on_bad_lines='skip', low_memory=False)
df['Crime Code'] = pd.to_numeric(df['Crime Code'], errors='coerce')
df.dropna(subset=['Crime Code'], inplace=True)
df['Crime Code'] = df['Crime Code'].astype(int)

# Encode Area Name
df['Area Name'] = df['Area Name'].fillna('Unknown')
le = LabelEncoder()
df['Area Name Encoded'] = le.fit_transform(df['Area Name'])

# Save the mapping for decoding later
crime_mapping = df.groupby('Crime Code')['Area Name'].agg(lambda x: pd.Series.mode(x)[0]).to_dict()
with open('crime_mapping.pkl', 'wb') as f:
    pickle.dump(crime_mapping, f)

# Features and target
X = df[['Area Name Encoded']]
y = df['Crime Code']

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open('rf_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Random Forest model saved as rf_model.pkl")
