import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load data
df = pd.read_csv("data/house_data.csv")

# Encode location
le = LabelEncoder()
df['location'] = le.fit_transform(df['location'])

# Features & Target
X = df.drop("price", axis=1)
y = df["price"]

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le, open("encoder.pkl", "wb"))

print("Model Trained & Saved Successfully!")