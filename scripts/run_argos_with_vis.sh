#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 2 ]; then
    echo "Assumes to be called from scripts directory"
    echo "Usage: $0 <input_file> <output_file> [<btree>...]"
    echo "e.g. './run_argos_with_vis.sh ../irace_experiment/experiments-folder/aggregation.argos /tmp/vis.argos  --nroot 3 --nchildroot 2 --n0 6 --c0 1 --p0 0.79 --n1 0 --nchild1 3 --n10 5 --a10 7 --p10 0.5 --n11 6 --c11 2 --p11 0.32 --n12 6 --c12 5 --p12 0.29'"
    exit 1
fi

# Check for the --no-vis flag
NOVIS=""
if [[ "$1" == "--no-vis" ]]; then
    # Special code to execute if --no-vis is present
    echo "Running in no-visualization mode."
    NOVIS="$1"
    # Add your special code here
    shift  # Remove the --no-vis flag from the arguments
fi

# Assign input and output file names from command-line arguments
input_file="$1"
output_file="$2"
shift 2  # Shift the first two arguments, so $@ now contains all remaining arguments
echo $input_file $output_file $@
if [ -z "$NOVIS" ]; then
    echo "Running in visualization mode."
    ./remove_vis_comment.sh "$input_file" "$output_file"
else
output_file=$input_file
fi

# Check if the previous script was successful
if [ $? -ne 0 ]; then
    echo "Failed to remove comment from the input file."
    exit 1
fi

# Run the podman command with all remaining arguments
podman run --rm -it \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    -v "$output_file:/root/aac.argos" \
    --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
    --privileged automode AutoMoDe/bin/automode_main_bt -c /root/aac.argos --bt-config "$@"

# docker run --rm -it --privileged -e DISPLAY -v "$output_file:/root/aac.argos" --volume="/tmp/.X11-unix:/tmp/.X11-unix:ro" automode AutoMoDe/bin/automode_main_bt -c /root/aac.argos --bt-config "$@"

# Check if the podman command was successful
if [ $? -ne 0 ]; then
    echo "Podman commands failed."
    exit 1
fi

echo "Podman commands executed successfully."
