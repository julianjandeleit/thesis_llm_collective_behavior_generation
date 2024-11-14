#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_file> <output_file>"
    exit 1
fi

# Assign input and output file names from command-line arguments
input_file="$1"
output_file="$2"

# Use sed to remove the comment around the qt-opengl node
sed '/<!--.*<qt-opengl>/,/<\/qt-opengl>-->/ {
    s/<!--//; 
    s/-->//;
}' "$input_file" > "$output_file"

echo "Comment around qt-opengl removed and saved to $output_file"
