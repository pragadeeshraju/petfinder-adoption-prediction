import unittest  
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, recall_score
from sklearn.preprocessing import LabelEncoder
from predict import make_predictions, load_model

# Read data from the given URL and load it into a Pandas DataFrame
GS_URL = "gs://cloud-samples-data/ai-platform-unified/datasets/tabular/petfinder-tabular-classification.csv"
df = pd.read_csv(GS_URL)

# Use LabelEncoder to convert the target variable 'Adopted' to binary values (0 and 1)
label_encoder = LabelEncoder()
df['Adopted'] = label_encoder.fit_transform(df['Adopted'])

# Define a test class for prediction functions
class TestPrediction(unittest.TestCase):
    # Test the prediction function 
    def test_make_predictions(self):
        model = load_model()
        input_data = df.drop(columns=['Adopted'])
        predictions = make_predictions(input_data, model)
        f1 = f1_score(df['Adopted'], predictions, pos_label=1)
        accuracy = accuracy_score(df['Adopted'], predictions)
        recall = recall_score(df['Adopted'], predictions, pos_label=1)

        self.assertGreater(f1, 0.8, "F1 Score Failed (<= 0.8)")
        self.assertGreater(accuracy, 0.7, "Accuracy Failed (<= 0.7)")
        self.assertGreater(recall, 0.9, "Recall Failed (<= 0.9)")        

if __name__ == '__main__':  
    unittest.main()