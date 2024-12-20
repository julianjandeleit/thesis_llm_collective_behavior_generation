#!/bin/bash

# Function to display help
function display_help() {
    echo "Usage: $0 [--llmtree] <pickle_file> <number>"
    echo
    echo "Options:"
    echo "  --llmtree       Use 'llm_behavior_tree: ' to extract the behavior tree."
    echo "  --help          Display this help message."
    echo
    echo "Example:"
    echo "  $0 --llmtree ../ressources/automode_evaluated_seed14_n300_24-12-15.pickle 075"
    echo "  $0 ../ressources/automode_evaluated_seed14_n300_24-12-15.pickle 075"
    exit 0
}

# Check for help option
for arg in "$@"; do
    if [[ "$arg" == "--help" ]]; then
        display_help
    fi
done

# Check if the correct number of arguments is provided
if [ "$#" -lt 2 ]; then
    echo "Error: Insufficient arguments."
    display_help
fi

# Initialize variables
USE_LLM=false
PICKLE_FILE=""
NUMBER=""

# Parse arguments
for arg in "$@"; do
    if [[ "$arg" == "--llmtree" ]]; then
        USE_LLM=true
    elif [[ -z "$PICKLE_FILE" ]]; then
        PICKLE_FILE="$arg"
    elif [[ -z "$NUMBER" ]]; then
        NUMBER="$arg"
    fi
done

# Check if both pickle file and number are provided
if [ -z "$PICKLE_FILE" ] || [ -z "$NUMBER" ]; then
    echo "Error: Pickle file and number must be specified."
    display_help
fi

EXTRACTED_DIR="/tmp/extracted"
VIS_DIR="/tmp/vis"

# Execute the first command and capture the output
OUTPUT=$(python extract_argos.py "$PICKLE_FILE" "$NUMBER" "$EXTRACTED_DIR")

# Check if the command was successful
if [ $? -ne 0 ]; then
    echo "Error executing extract_argos.py"
    echo "$OUTPUT"
    exit 1
fi

# Determine the prefix to use for extracting the behavior tree
if $USE_LLM; then
    PREFIX="llm_behavior_tree: "
else
    PREFIX="behavior_tree: "
fi

# Extract the behavior tree from the output
#BEHAVIOR_TREE=$(echo "$OUTPUT" | grep -oP "${PREFIX}\K.*")
BEHAVIOR_TREE=$(echo "$OUTPUT" | grep -oP "${PREFIX}\K.*" | head -n 1 | tr -d '\n')

# Check if the behavior tree was found
if [ -z "$BEHAVIOR_TREE" ]; then
    echo "Behavior tree not found in output."
    exit 1
fi
echo using $PREFIX $BEHAVIOR_TREE
# Execute the second command with the extracted behavior tree
./run_argos_with_vis.sh "$EXTRACTED_DIR" "$VIS_DIR" $BEHAVIOR_TREE > /dev/null

# Check if the second command was successful
if [ $? -ne 0 ]; then
    echo "Error executing run_argos_with_vis.sh"
    exit 1
fi

echo "Commands executed successfully."
