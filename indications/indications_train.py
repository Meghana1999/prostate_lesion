import pandas as pd
import numpy as np
import re 
import string 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

header = "/mnt/storage/RAD_PATH/lesion/prostate_lesion/"
header1 = "/mnt/storage/RAD_PATH/clinical_indications/"

class ProstateIndicationClassifier:
    def __init__(self):
        self.xgb_model = {}
        self.vectorizer = TfidfVectorizer(ngram_range=(2, 2))
        self.labels = ['pca', 'no prior bx', 'benign bx', 'xrt planning', 'recurrence', 'as']
        self.y_map = {'pca': 0, 'no prior bx': 1, 'benign bx': 2, 'xrt planning': 3, 'recurrence': 4, 'as': 5}
        self.LMmodel = xgb.XGBClassifier(objective='multi:softmax')
        
    def model_load(self, modelpath):
        self.LMmodel = xgb.XGBClassifier()
        self.LMmodel.load_model(modelpath + 'CIXgboost.json')
        self.vectorizer = joblib.load(modelpath+'tfidf_vectorizer.pkl')   
        print('Model loaded!!') 

    def model_save(self, modelpath):
        try:
            self.xgb_model.save_model(modelpath + 'CIXgboost.json') 
            joblib.dump(self.vectorizer, modelpath+'tfidf_vectorizer.pkl')   
            print('Model saved!!')
        except:
            print('Model saving didn\'t work') 

    def train_main(self, traindf, testdf):
        print('Train main execution started') 
        traindf.dropna(inplace=True)
        traindf.rename(columns={'Prostate_Indications': 'Comments', 'NELLY INDICATIONS': 'classes'}, inplace=True)   
        print(traindf['Comments'])
        traindf['Comments'] = traindf['Comments'].astype(str)
        traindf = traindf.reset_index(drop=True)
        testdf.dropna(inplace=True)
        testdf.rename(columns={'Prostate_Indications': 'Comments', 'NELLY INDICATIONS': 'classes'}, inplace=True)   
        print(testdf['Comments'])
        testdf['Comments'] = testdf['Comments'].astype(str)
        testdf = testdf.reset_index(drop=True)
        X_train = traindf.Comments
        y_train = traindf.classes.map(self.y_map)
        X_test = testdf.Comments
        y_test = testdf.classes.map(self.y_map)
        
        self.vectorizer.fit(X_train)
        # joblib.dump(self.vectorizer, 'tfidf_vectorizer.pkl')  # Save the fitted vectorizer
        train_sentence_embeddings = self.vectorizer.transform(X_train).toarray()
        test_sentence_embeddings = self.vectorizer.transform(X_test).toarray()
        
        from sklearn.utils import compute_sample_weight
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
        self.xgb_model = xgb.XGBClassifier(objective='multi:softmax')
        self.xgb_model.fit(train_sentence_embeddings, y_train, sample_weight=sample_weights) 
        y_pred = self.xgb_model.predict(test_sentence_embeddings)  
        print('Test accuracy is {}'.format(accuracy_score(y_test, y_pred)))
        print(classification_report(y_test, y_pred))
        cm = confusion_matrix(y_test, y_pred) 
        sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', xticklabels=self.labels, yticklabels=self.labels) 
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix') 
        plt.xticks(rotation=45)
        plt.yticks(rotation=0) 
        plt.show()
        misclassified_data = pd.DataFrame({
            'Comments': testdf['Comments'],
            'True_label': testdf['classes'],
            'Predicted_label': y_pred
        })
        misclassified_data_filtered = misclassified_data[misclassified_data['True_label'] != misclassified_data['Predicted_label']]
        y_map = {0: 'pca', 1: 'no prior bx', 2: 'benign bx', 3: 'xrt planning', 4: 'recurrence', 5: 'as'}
        misclassified_data_filtered['classes'] = misclassified_data_filtered['True_label'].map(y_map)
        misclassified_data_filtered['Predicted_class'] = misclassified_data_filtered['Predicted_label'].map(y_map)
        misclassified_data_filtered.to_excel(header + "indications/output/misclassified_train_data.xlsx")

    def test_main(self, testdf, modelpath):
        print('Test main execution started')
        testdf.dropna(inplace=True)
        testdf.rename(columns={'Prostate_Indications': 'Comments', 'NELLY INDICATIONS': 'classes'}, inplace=True)
        testdf['Comments'] = testdf['Comments'].astype(str)
        testdf = testdf.reset_index(drop=True)
        X_test = testdf.Comments
        y_test = testdf.classes.map(self.y_map)
        print(testdf) 
         
        test_sentence_embeddings = self.vectorizer.transform(X_test).toarray()
        print(test_sentence_embeddings.shape)
        y_pred = self.LMmodel.predict(test_sentence_embeddings)
         
        misclassified_data = pd.DataFrame({
            'Comments': testdf['Comments'],
            'True_label': testdf['classes'],
            'Predicted_label': y_pred
        })
        misclassified_data_filtered = misclassified_data[misclassified_data['True_label'] != misclassified_data['Predicted_label']]
        y_map = {0: 'pca', 1: 'no prior bx', 2: 'benign bx', 3: 'xrt planning', 4: 'recurrence', 5: 'as'}
        misclassified_data_filtered['classes'] = misclassified_data_filtered['True_label'].map(y_map)
        misclassified_data_filtered['Predicted_class'] = misclassified_data_filtered['Predicted_label'].map(y_map)
        misclassified_data_filtered.to_excel(header + "indications/output/misclassified_test_data.xlsx")
        print('Prediction completed!')

    def preprocessing(self, x):
        text = x.lower()
        exclist = string.punctuation
        table_ = str.maketrans('', '', exclist)
        text = text.translate(table_)
        words = text.split()
        return ' '.join(words)