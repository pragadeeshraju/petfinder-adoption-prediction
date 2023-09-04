import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce
import os
import datetime


# Read data from the given URL and load it into a Pandas DataFrame
GS_URL = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
df = pd.read_csv(GS_URL)

# Use LabelEncoder to convert the target variable 'Adopted' to binary values (0 and 1).
label_encoder = LabelEncoder()
df['Adopted'] = label_encoder.fit_transform(df['Adopted'])

X = df.drop(columns=['Adopted'])
y = df['Adopted']

#split the dataset into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=23)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=23)

# Perform category encoding using OrdinalEncoder
encoder = ce.OrdinalEncoder()
X_train_encoded = encoder.fit_transform(X_train)
X_val_encoded = encoder.transform(X_val)
X_test_encoded = encoder.transform(X_test)

# Train an XGBoost model
model = xgb.XGBClassifier(early_stopping_rounds=10, eval_metric=["error", "logloss"])
model.fit(
    X_train_encoded,
    y_train,
    eval_set=[(X_val_encoded, y_val)],
    verbose=True,
)

results = model.evals_result()
print(results)

# Evaluate the model on the test set
y_pred = model.predict(X_test_encoded)
f1 = f1_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Log the metrics
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_file_name = f"log_{timestamp}.txt"
log_directory = "artifacts/logs/"
os.makedirs(log_directory, exist_ok=True)
log_file_path = os.path.join(log_directory, log_file_name)

log_message = f"{timestamp} - F1 Score: {f1}, Accuracy: {accuracy}, Recall: {recall}\n"
with open(log_file_path, "a") as log_file:
    log_file.write(log_message)

print(log_message)

# Save the trained model
model_directory = "artifacts/model"
os.makedirs(model_directory, exist_ok=True)
model.save_model(os.path.join(model_directory, "xgboost_model.model"))

print(f"Model saved to '{model_directory}'")
