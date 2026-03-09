import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import hamming_loss, f1_score

print("\n--- Urban Soundscape ML Training Started ---\n")

# -------------------------------
# Load CLEAN dataset
# -------------------------------
data = pd.read_csv("../datasets/processed/sonyc_clean.csv")

# Separate labels
label_cols = [c for c in data.columns if c.endswith("_presence")]
# Save label names for API
with open("../models/label_names.txt", "w") as f:
    for c in label_cols:
        f.write(c + "\n")

print("Saved label names to ../models/label_names.txt")


X = data.drop(columns=["audio_filename"] + label_cols)
y = data[label_cols]

print("Features:", X.shape)
print("Labels:", y.shape)

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# Model
# -------------------------------
model = MultiOutputClassifier(
    RandomForestClassifier(
        n_estimators=150,
        max_depth=20,
        n_jobs=-1,
        random_state=42,
        verbose=1
    )
)

print("\nTraining model...")
model.fit(X_train, y_train)

# -------------------------------
# Evaluation
# -------------------------------
y_pred = model.predict(X_test)

print("\n--- Performance ---")
print("Hamming Loss:", hamming_loss(y_test, y_pred))
print("Micro F1:", f1_score(y_test, y_pred, average="micro"))
print("Macro F1:", f1_score(y_test, y_pred, average="macro"))

# -------------------------------
# Save model
# -------------------------------
joblib.dump(model, "../models/sound_rf_multilabel.pkl")
print("\nModel saved to ../models/sound_rf_multilabel.pkl")

