import sys
import pandas as pd
import argparse
from train import PathologyDataProcessor

def main(input_file, output_file_path_res_1, output_file_path_res_2):
    try:
        df = pd.read_excel(input_file)
        processor = PathologyDataProcessor(df)

        # Function calls
        results_df = processor.process_dataframe_bx1()
        results_df2 = processor.process_dataframe_bx2()

        # Saving the results
        results_df.to_excel(output_file_path_res_1, index=False)
        results_df2.to_excel(output_file_path_res_2, index=False)

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process pathology data.")
    parser.add_argument("input_file", help="Path to the input Excel file")
    parser.add_argument("output_file_path_res_1", help="Path to the output Excel file for PATH_RES_1 results")
    parser.add_argument("output_file_path_res_2", help="Path to the output Excel file for PATH_RES_2 results")
    
    args = parser.parse_args()

    main(args.input_file, args.output_file_path_res_1, args.output_file_path_res_2)

 