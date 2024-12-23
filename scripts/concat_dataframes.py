import pandas as pd
import argparse

def concatenate_dataframes(file1, file2, output_file):
    # Read the two DataFrames from pickle files
    df1 = pd.read_pickle(file1)
    df2 = pd.read_pickle(file2)

    # Concatenate the DataFrames
    concatenated_df = pd.concat([df1, df2], ignore_index=True)

    # Reset the index
    concatenated_df.reset_index(drop=True, inplace=True)

    # Write the result to a new pickle file
    concatenated_df.to_pickle(output_file)
    print(f"Concatenated DataFrame saved to {output_file}")

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Concatenate two DataFrames from pickle files.')
    parser.add_argument('file1', type=str, help='Path to the first pickle file')
    parser.add_argument('file2', type=str, help='Path to the second pickle file')
    parser.add_argument('output_file', type=str, help='Path to the output pickle file')

    # Parse the arguments
    args = parser.parse_args()

    # Call the function with the provided arguments
    concatenate_dataframes(args.file1, args.file2, args.output_file)
