import cv2
import numpy as np
from PIL import Image
import argparse

def process_gif(input_gif, output_image, decay_rate=5.0):
    # Load GIF and convert to frames
    gif = Image.open(input_gif)
    frames = []
    try:
        while True:
            frames.append(np.array(gif.copy()))
            gif.seek(gif.tell() + 1)
    except EOFError:
        pass

    # Convert frames to RGB
    frames = [cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB) for frame in frames]

    # Create a blank image to accumulate traces
    trace_image = np.zeros_like(frames[0], dtype=np.float32)

    # Process each frame
    num_frames = len(frames)
    for i in range(1, num_frames):
        # Calculate the absolute difference between consecutive frames
        diff = cv2.absdiff(frames[i], frames[i - 1])
        
        # Convert to grayscale
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY)
        
        # Threshold the difference to create a binary mask
        _, mask = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)
        
        # Calculate opacity with reversed exponential decay:
        # Strongest (opacity 1) for last frames, lower for earlier ones.
        opacity = np.exp(-decay_rate * ((num_frames - i) / num_frames))
        
        # Update the trace image with the moving pixels, applying the opacity
        trace_image[mask == 255] += np.array([255, 255, 255]) * opacity

    # Normalize the trace image to the range [0, 255]
    trace_image = np.clip(trace_image, 0, 255).astype(np.uint8)

    # Save the trace image
    if not cv2.imwrite(output_image, trace_image):
        print(f"Error: Could not save the image to {output_image}. Check the file extension and format.")
    else:
        print(f"Trace image saved as: {output_image}")

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process a GIF to create a trace image of moving objects.")
    parser.add_argument("input_gif", type=str, help="Path to the input GIF file.")
    parser.add_argument("output_image", type=str, help="Path to save the output trace image.")
    parser.add_argument("--decay_rate", type=float, default=5.0, help="Rate of exponential decay for opacity.")

    # Parse the arguments
    args = parser.parse_args()

    # Process the GIF
    process_gif(args.input_gif, args.output_image, args.decay_rate)
