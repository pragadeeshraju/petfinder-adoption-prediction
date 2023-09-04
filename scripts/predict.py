import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import category_encoders as ce

GS_URL = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
df = pd.read_csv(GS_URL)

# Use LabelEncoder to convert the target variable 'Adopted' to binary values (0 and 1)
label_encoder = LabelEncoder()
df['Adopted'] = label_encoder.fit_transform(df['Adopted'])

def load_model():
    model = xgb.XGBClassifier()
    model.load_model("artifacts/model/xgboost_model.model")
    return model

model = load_model()

# Function to make predictions 
def make_predictions(input_data, model):
    # Perform category encoding using OrdinalEncoder
    encoder = ce.OrdinalEncoder()
    input_data_encoded = encoder.fit_transform(input_data)
    
    # Make predictions
    predictions = model.predict(input_data_encoded)
    return predictions

predictions = make_predictions(df.drop(columns=['Adopted']), model)
# Inverse transform the predictions to get 'Yes' or 'No'
pred = label_encoder.inverse_transform(predictions)
df['Adopted_prediction'] = pred

# Save the output to 'output/results.csv'
output_file = 'output1/results.csv'
df.to_csv(output_file, index=False)