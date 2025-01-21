import argparse
import pickle
import pandas as pd

# Define the function to convert to SFT format
def convert_to_sft_format(row):
    # Compare scores and assign values accordingly
    if row["scores_bt1"] > row["scores_bt2"]:
        row["avg_score"] = row["scores_bt2"]
        row["behavior_tree"] = row["llmout_B"]
        row["llm_avg_score"] = row["scores_bt1"]
        row["llm_behavior_tree"] = row["llmout_A"]
    else:
        row["avg_score"] = row["scores_bt1"]
        row["behavior_tree"] = row["llmout_A"]
        row["llm_avg_score"] = row["scores_bt2"]
        row["llm_behavior_tree"] = row["llmout_B"]
    
    # Assign the description
    row["description"] = row["llmin"]
    
    return row

def main(input_file, output_file):
    # Load the original DataFrame from the input pickle file
    with open(input_file, "rb") as file:
        original_df = pickle.load(file)

    # Apply the conversion function to each row of the DataFrame
    df = original_df.apply(convert_to_sft_format, axis=1)

    # Save the modified DataFrame to the output pickle file
    with open(output_file, "wb") as file:
        pickle.dump(df, file)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Convert DataFrame to SFT format. better score is converted to llm column")
    parser.add_argument("input_file", type=str, help="Path to the input pickle file.")
    parser.add_argument("output_file", type=str, help="Path to the output pickle file.")

    # Parse the arguments
    args = parser.parse_args()

    # Call the main function with the provided arguments
    main(args.input_file, args.output_file)
