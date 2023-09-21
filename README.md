# prostate_lesion_Extraction

This contains the code to extract the Lesion paragraphs, No.of lesions, Volume, Scores etc from Prostate Clinical Indications file

## How to Run the Code

1.  python prostate_main.py --input-file "Path to input file" --output-file "path to output file"


# Prostate Lesion Indication Classifier

This repository contains the code for a multi-class classification model that predicts the indication of prostate lesions based on clinical text data. The model is trained using the XGBoost algorithm and uses TF-IDF vectorization for feature extraction from the text data.

## Files in this Repository

- `indications_train.py`: This script contains the `ProstateIndicationClassifier` class which is used for training the model. The class includes methods for data preprocessing, model training, and model evaluation.

- `indications_main.py`: This is the main script that calls the methods from the `ProstateIndicationClassifier` class for either training or testing the model based on the mode specified.

## How to Run the Code

1. Clone this repository to your local machine using `git clone https://github.com/Meghana1999/prostate_lesion.git`.


2. To train the model, run the following command:
    python indications_main.py --flag Train --traindata path_to_train_data --testdata path_to_test_data --modelpath path_to_save_model  


3. To test the model, run the following command:

    
    python indications_main.py --flag Test --testdata path_to_test_data --modelpath path_to_load_model
    

## Dependencies

This code requires the following libraries:

- pandas
- numpy
- re
- string
- sklearn
- xgboost
- matplotlib
- seaborn

You can install these libraries using pip:
pip install pandas numpy sklearn xgboost matplotlib seaborn

There is a requirements.txt file 
## Contact

If you have any questions or run into any issues, please open an issue on this GitHub repository.