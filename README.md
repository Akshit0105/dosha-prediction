# create_app.py
app_code = """
import streamlit as st
import pandas as pd
import pickle
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load & merge data
def load_and_merge_data():
    df1 = pd.read_csv("data/ayurvedic_dosha_dataset.csv")
    df2 = pd.read_csv("data/data.csv")
    df3 = pd.read_csv("data/Updated_Prakriti_With_Features.csv")

    for df in [df1, df2, df3]:
        df.columns = df.columns.str.strip()
        for col in df.columns:
            if col.lower() in ["dosha", "prakriti", "type"]:
                df.rename(columns={col: "Dosha"}, inplace=True)

    df = pd.concat([df1, df2, df3], ignore_index=True, sort=False)
    df = df.dropna(subset=["Dosha"])
    df = df.fillna("Unknown")
    return df

# Train model if not exists
def train_model():
    df = load_and_merge_data()
    X = df.drop("Dosha", axis=1)
    y = df["Dosha"]

    le_dict = {}
    for col in X.columns:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        le_dict[col] = le

    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)

    pickle.dump(model, open("dosha_model.pkl", "wb"))
    pickle.dump(target_encoder, open("target_encoder.pkl", "wb"))
    pickle.dump(le_dict, open("feature_encoders.pkl", "wb"))

if not os.path.exists("dosha_model.pkl"):
    train_model()

model = pickle.load(open("dosha_model.pkl", "rb"))
target_encoder = pickle.load(open("target_encoder.pkl", "rb"))
le_dict = pickle.load(open("feature_encoders.pkl", "rb"))

st.title("🧘 Dosha Prediction & Risk Assessment")

st.subheader("Enter Your Details")
user_input = {}
for feature in le_dict.keys():
    options = le_dict[feature].classes_
    user_input[feature] = st.selectbox(feature, options)

input_df = pd.DataFrame([user_input])

encoded_input = input_df.copy()
for col in encoded_input.columns:
    encoded_input[col] = le_dict[col].transform(encoded_input[col].astype(str))

def dosha_solution(dosha):
    if dosha == "Vata":
        return ["Eat warm, cooked food", "Follow a routine lifestyle", "Avoid cold environments"]
    elif dosha == "Pitta":
        return ["Avoid spicy & oily food", "Stay cool", "Reduce stress"]
    elif dosha == "Kapha":
        return ["Exercise regularly", "Avoid oily/heavy food", "Stay active"]
    else:
        return ["Maintain a balanced lifestyle"]

if st.button("Predict Doshas & Risks"):
    probs = model.predict_proba(encoded_input)[0]
    labels = target_encoder.classes_

    prob_df = pd.DataFrame({
        "Dosha": labels,
        "Probability": probs,
        "Risk (%)": [(1 - p) * 100 for p in probs]
    }).sort_values("Probability", ascending=False)

    st.subheader("Predicted Dosha Probabilities")
    st.bar_chart(prob_df.set_index("Dosha")["Probability"])

    st.subheader("Risk Levels for Each Dosha")
    st.bar_chart(prob_df.set_index("Dosha")["Risk (%)"])

    st.subheader("Tips to Minimize Risk")
    for i, row in prob_df.iterrows():
        st.write(f"**{row['Dosha']}** (Risk: {row['Risk (%)']:.2f}%):")
        for tip in dosha_solution(row['Dosha']):
            st.write("•", tip)
        st.write("---")
"""

# Write the app code to app.py
with open("app.py", "w") as f:
    f.write(app_code)

print("✅ app.py created successfully!")
