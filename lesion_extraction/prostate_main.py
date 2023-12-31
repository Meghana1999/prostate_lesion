import argparse
import prostate_train
from prostate_train import ProstateDataProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Prostate DataExport file")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input Excel file")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output Excel file")
    args = parser.parse_args() 
    
    data_processor = ProstateDataProcessor()
    data_processor.process_prostate_data(args.input_file, args.output_file)
    print('CODE EXECUTION COMPLETED')


#### To Run ###
## python prostate_main.py --input-file input_file.xlsx --output-file output_file.xlsx

