import pandas as pd
import os
import argparse
import sys
import indications_train as train 
 
parser = argparse.ArgumentParser()
parser.add_argument('--modelpath', type=str,
                    default=header+"indications/savedModel/", 
                    help='path to the trained model')

parser.add_argument('--traindata', type=str, default= header1+"prostate_train.xlsx",
                    help='path to the train')
 
parser.add_argument('--testdata', type=str, default= header1+"prostate_test.xlsx", 
                    help='path to the test file')

parser.add_argument('--flag', type=str, default='Test', 
                    help='flag to signify the mode of model use -  Train/Test')

args = parser.parse_args()


def main():
    clinical_indications = train.ProstateIndicationClassifier() 
    if args.flag == 'Train':
        traindf = pd.read_excel(args.traindata) 
        testdf = pd.read_excel(args.testdata) 
        clinical_indications.train_main(traindf, testdf)
        clinical_indications.model_save(args.modelpath) 
            
    if args.flag == 'Test':
        testdf = pd.read_excel(args.testdata)
        clinical_indications.model_load(args.modelpath)
        final_test = clinical_indications.test_main(testdf,args.modelpath)
        print('FINISHED EXECUTION') 

if __name__ == "__main__":
    main()