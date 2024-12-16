import pandas as pd
import argparse
import os

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract argos field from DataFrame at specified index.')
    parser.add_argument('input_path', type=str, help='Path to the input pickle file containing the DataFrame.')
    parser.add_argument('index', type=int, help='Index of the row to extract the argos field from.')
    parser.add_argument('output_path', type=str, help='Path to the output file where the argos field will be written.')

    # Parse arguments
    args = parser.parse_args()

    # Read the DataFrame from the pickle file
    try:
        df = pd.read_pickle(args.input_path)
    except Exception as e:
        print(f"Error reading the pickle file: {e}")
        return

    # Check if the index is valid
    if args.index < 0 or args.index >= len(df):
        print(f"Index {args.index} is out of bounds for the DataFrame with {len(df)} rows.")
        return

    # Extract the 'argos' field from the specified index
    argos_value = df.at[args.index, 'argos']

    # Write the argos value to the output file
    try:
        with open(args.output_path, 'w') as f:
            f.write(str(argos_value))
        #print(f"Successfully wrote argos value to {args.output_path}")
    except Exception as e:
        print(f"Error writing to the output file: {e}")
        
    print(f"avg_score: {df.iloc[args.index].avg_score}")
    print(f"description: {df.iloc[args.index].description}")
    print(f"behavior_tree: {df.iloc[args.index].behavior_tree}")
    print(f"llm_behavior_tree: {df.iloc[args.index].llm_behavior_tree if 'llm_behavior_tree' in df.keys() else None}" )

if __name__ == '__main__':
    main()
