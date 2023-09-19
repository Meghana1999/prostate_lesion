import argparse
from prostate_train import ProstateDataProcessor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Prostate DataExport file")
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input Excel file")
    parser.add_argument("--output-file", type=str, required=True, help="Path to the output Excel file")
    args = parser.parse_args()
    
    # Create an instance of the ProstateDataProcessor class
    data_processor = ProstateDataProcessor()
    data_processor.process_prostate_data(args.input_file, args.output_file)
    print('CODE EXECUTION COMPLETED')
