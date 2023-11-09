import pandas as pd
import numpy as np
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import xgboost as xgb

class ProstateIndicationClassifier:
    def __init__(self):
        self.xgb_model = xgb.XGBClassifier(objective='multi:softmax')
        self.vectorizer = TfidfVectorizer(ngram_range=(2, 2))
        self.labels = ['pca', 'no prior bx', 'benign bx', 'xrt planning', 'recurrence', 'as']
        self.y_map = {'pca': 0, 'no prior bx': 1, 'benign bx': 2, 'xrt planning': 3, 'recurrence': 4, 'as': 5}

    def extract_and_preprocess(self, df):
        df['Prostate_Indications'] = df['NARRATIVE'].apply(self.extract_prostate_indications)
        df['Comments'] = df['Prostate_Indications'].apply(self.preprocessing)
        df = df[['Comments', 'NELLY INDICATIONS']].dropna()
        df['NELLY INDICATIONS'] = df['NELLY INDICATIONS'].map(self.y_map)
        return df

    def train_main(self, traindf, testdf):
        print('Train main execution started')

        traindf_processed = self.extract_and_preprocess(traindf)
        testdf_processed = self.extract_and_preprocess(testdf)

        # Transform the comments into TF-IDF features
        self.vectorizer.fit(traindf_processed['Comments'])
        train_sentence_embeddings = self.vectorizer.transform(traindf_processed['Comments']).toarray()
        test_sentence_embeddings = self.vectorizer.transform(testdf_processed['Comments']).toarray()

        # Balance the classes
        from sklearn.utils import compute_sample_weight
        sample_weights = compute_sample_weight(class_weight='balanced', y=traindf_processed['NELLY INDICATIONS'])

        # Fit the model
        self.xgb_model.fit(train_sentence_embeddings, traindf_processed['NELLY INDICATIONS'], sample_weight=sample_weights)

        # Predict on the test data
        y_pred = self.xgb_model.predict(test_sentence_embeddings)

        # Output the results
        print('Test accuracy is {}'.format(accuracy_score(testdf_processed['NELLY INDICATIONS'], y_pred)))
        print(classification_report(testdf_processed['NELLY INDICATIONS'], y_pred))
        cm = confusion_matrix(testdf_processed['NELLY INDICATIONS'], y_pred)
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=self.labels, yticklabels=self.labels)
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.show()

        print('Training completed.')
    
    def extract_and_preprocess_test_data(self, df):
        df['Prostate_Indications'] = df['NARRATIVE'].apply(self.extract_prostate_indications)
        df['Comments'] = df['Prostate_Indications'].apply(self.preprocessing)
        df = df[['Comments']].dropna()
        return df
        
        
    def test_main(self, testdf, model_path):
        print('Test main execution started') 
        # Load the model
        self.model_load(model_path)  
        testdf_processed = self.extract_and_preprocess_test_data(testdf)
        test_sentence_embeddings = self.vectorizer.transform(testdf_processed['Comments']).toarray() 
        # Predict on the test data
        y_pred = self.xgb_model.predict(test_sentence_embeddings) 
        # DataFrame with comments and predicted labels
        predictions = pd.DataFrame({
            'Comments': testdf_processed['Comments'],
            'PredictedLabel': [self.labels[pred] for pred in y_pred]
        })
        print('Testing completed.')
        return predictions

    def model_save(self, model_path):
        # Save the model and the vectorizer
        joblib.dump(self.xgb_model, model_path + 'xgb_model.pkl')
        joblib.dump(self.vectorizer, model_path + 'tfidf_vectorizer.pkl')
        print(f"Model saved to {model_path}")

    def model_load(self, model_path):
        # Load the model and the vectorizer
        self.xgb_model = joblib.load(model_path + 'xgb_model.pkl')
        self.vectorizer = joblib.load(model_path + 'tfidf_vectorizer.pkl')
        print(f"Model loaded from {model_path}")

    def preprocessing(self, x):
        text = x.lower()
        text = re.sub(r'\d+', '', text)  
        text = text.translate(str.maketrans('', '', string.punctuation))   
        text = text.strip()
        return text
 
    def extract_prostate_indications(text): 
        pattern = r'(?:HISTORY:|INFORMATION:|History:)(.*?)(?=(PROSTATE|COMPARISON|TECHNIQUE))' 
        matches = re.findall(pattern, text, flags=re.DOTALL) 
        if matches: 
            result = matches[0] 
            if isinstance(result, tuple):
                result = result[0] 
            return result.strip()
        else: 
            return ''

 
