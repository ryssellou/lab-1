import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

# Load models from .h5 files
readability_model = load_model("readability_model_fixed_2.h5")
bug_model = load_model("bug_model_fixed_2.h5")

# Define label encoders with known training labels
complexity_labels = ["Low", "Medium", "High"]
big_o_labels = ["O(1)", "O(log n)", "O(n)", "O(n log n)", "O(n^2)", "O(2^n)", "O(n!)"]

big_o_mapping = {
    "O(n!)": "Horrible and Explosive", "O(2^n)": "Horrible and Explosive", "O(n^2)": "Bad and Inefficient",
    "O(n log n)": "Fair and Moderate", "O(n)": "Good and Scalable", "O(log n)": "Excellent and Efficient", "O(1)": "Excellent and Efficient"
}

le_complexity = LabelEncoder().fit(complexity_labels)
le_big_o = LabelEncoder().fit(big_o_labels)

def analyze_code(code):
    complexity = len(code.split())  # Example complexity calculation
    big_o = np.random.randint(0, len(big_o_labels))  # Placeholder for Big-O analysis
    return complexity, big_o

def safe_inverse_transform(label_encoder, index):
    if isinstance(index, np.ndarray):
        index = index.item()
    if 0 <= index < len(label_encoder.classes_):
        return label_encoder.inverse_transform([index])[0]
    return "Unknown"

def predict_readability(code):
    ex_complexity, ex_big_o = analyze_code(code)
    ex_input = np.array([[ex_complexity / 1000]])
    preds = readability_model.predict(ex_input).squeeze()
    
    if preds.ndim == 1 and preds.shape[0] >= 3:
        big_o_pred = safe_inverse_transform(le_big_o, np.argmax(preds[1:4]))
        return {
            "Complexity": safe_inverse_transform(le_complexity, np.argmax(preds[0:3])),
            "Big-O": big_o_pred,
            "Big-O Label": big_o_mapping.get(big_o_pred, "Unknown"),
            "Readability": "Readable" if preds[3] < 0.5 else "Unreadable"
        }
    return {"Complexity": "Error", "Big-O": "Error", "Big-O Label": "Error", "Readability": "Error"}

def extract_features(code):
    features = np.array([
        len(code), code.count("{"), code.count("}"), code.count(";")
    ])
    return features, np.random.randint(0, len(big_o_labels))

def predict_bug_localization(code):
    try:
        ex_features, _ = extract_features(code)
        ex_features = np.array(ex_features, dtype=np.float32).reshape(1, -1)
        normalization_factors = np.array([1000, 50, 50, 50], dtype=np.float32)
        ex_input = ex_features / normalization_factors

        preds = bug_model.predict(ex_input).squeeze()

        if preds.ndim == 1 and preds.shape[0] >= 3:
            big_o_pred = safe_inverse_transform(le_big_o, np.argmax(preds[1:3]))
            return {
                "Complexity": safe_inverse_transform(le_complexity, np.argmax(preds[0:2])),
                "Big-O": big_o_pred,
                "Big-O Label": big_o_mapping.get(big_o_pred, "Unknown"),
                "Bug Presence": "Bug" if preds[2] > 0.5 else "No Bug"
            }
    except Exception:
        pass
    return {"Complexity": "Low", "Big-O": "O(1)", "Big-O Label": "Excellent and Efficient", "Bug Presence": "No Bug"}

st.title("Code Analysis App")
st.markdown("### Input your code below:")

code_input = st.text_area("Paste your code here...", height=250)

if st.button("Analyze Code"):
    if code_input.strip():
        readability_results = predict_readability(code_input)
        bug_results = predict_bug_localization(code_input)

        st.markdown("### Readability Model Predictions")
        st.write(f"**Predicted Complexity:** {readability_results['Complexity']}")
        st.write(f"**Predicted Big-O:** {readability_results['Big-O']}")
        st.write(f"**Predicted Big-O Label:** {readability_results['Big-O Label']}")
        st.write(f"**Predicted Readability:** {readability_results['Readability']}")

        st.markdown("### Bug Localization Model Predictions")
        st.write(f"**Predicted Complexity:** {bug_results['Complexity']}")
        st.write(f"**Predicted Big-O:** {bug_results['Big-O']}")
        st.write(f"**Predicted Big-O Label:** {bug_results['Big-O Label']}")
        st.write(f"**Predicted Bug Presence:** {bug_results['Bug Presence']}")
    else:
        st.warning("Please enter some code before analyzing.")
