import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression

DATA_FILE = "data.csv"
MODEL_FILE = "model.pkl"
COLUMNS_FILE = "columns.pkl"
ENCODERS_FILE = "encoders.pkl"

print("Loading data...")
df = pd.read_csv("data.csv", low_memory=False)
print(f"Data shape: {df.shape}")

# 🟦 Target variable
if 'loanstatus' not in df.columns:
    print("Error: 'loanstatus' column not found in data!")
    exit(1)

df['credit_risk'] = df['loanstatus'].apply(
    lambda x: 1 if x == 'CHGOFF' else 0
)

print(f"Class distribution:\n{df['credit_risk'].value_counts()}")

# 🟦 Feature selection
features = [
    'grossapproval',
    'terminmonths',
    'initialinterestrate',
    'businesstype',
    'naicsdescription',
    'jobssupported',
    'collateralind',
    'revolverstatus'
]

missing_features = [f for f in features if f not in df.columns]
if missing_features:
    print(f"Error: Missing features: {missing_features}")
    exit(1)

df = df[features + ['credit_risk']]

# 🟦 Handle missing values
print("Handling missing values...")
for col in df.columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].median())
    else:
        df[col] = df[col].fillna("Unknown")

print(f"Missing values after handling:\n{df.isnull().sum()}")

# 🟦 Encoding
print("Encoding categorical features...")
encoders = {}
categorical_cols = ['businesstype', 'naicsdescription', 'collateralind']

for col in categorical_cols:
    if col in df.columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        encoders[col] = encoder

# Save encoders
pickle.dump(encoders, open(ENCODERS_FILE, "wb"))

# 🟦 Features & target
X = df.drop('credit_risk', axis=1)
y = df['credit_risk']

# Save columns
pickle.dump(X.columns.tolist(), open(COLUMNS_FILE, "wb"))

print(f"Features for training: {X.columns.tolist()}")

# 🟦 Train-test split
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# 🟦 Logistic Regression (MAIN MODEL)
print("Training Logistic Regression model...")

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("✅ Model trained successfully!")

# 🟦 Evaluation
print("Evaluating model...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk']))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 🟦 Save model
print(f"Saving model to '{MODEL_FILE}'...")
pickle.dump(model, open(MODEL_FILE, "wb"))

print("✅ All files saved successfully!")