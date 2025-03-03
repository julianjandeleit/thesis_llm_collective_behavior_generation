import cv2
import numpy as np
import argparse
import imageio
from pathlib import Path

# in contras to gif_to_trace this only tracks one robot

def get_contours(frame, prev_contour=None, max_dist=50):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    if prev_contour is None:
        return contours
    
    closest_contour = None
    min_distance = float('inf')
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            prev_cx, prev_cy = np.mean(prev_contour[:, 0, :], axis=0)
            dist = np.sqrt((cx - prev_cx)**2 + (cy - prev_cy)**2)
            
            if dist < min_distance and dist < max_dist:
                min_distance = dist
                closest_contour = contour
    
    return closest_contour

def process_gif(gif_path, output_path, overlay_output, path_image_output, max_dist=50, index=0):
    gif = cv2.VideoCapture(gif_path)
    ret, first_frame = gif.read()
    if not ret:
        print("Error: Cannot read the first frame of the GIF")
        return
    
    height, width, _ = first_frame.shape
    trace_img = np.zeros((height, width, 3), dtype=np.uint8)
    contours = get_contours(first_frame)
    if not contours or index >= len(contours):
        print("Error: No contours found or invalid index.")
        return
    prev_contour = contours[index]
    frames = []
    tracking = True
    
    prev_frame = first_frame.copy()
    history = []
    
    while True:
        ret, frame = gif.read()
        if not ret:
            break
        
        if tracking:
            diff = cv2.absdiff(frame, prev_frame)
            prev_frame = frame.copy()
            contour = get_contours(diff, prev_contour, max_dist)
            
            if contour is not None:
                history.append(contour)
                prev_contour = contour
            else:
                tracking = False  # Stop tracking if no contour is found
        
        trace_img.fill(0)
        for i, contour in enumerate(history):
            gray_level = int(255 - (i / len(history)) * 200)  # Gradually turning gray
            color = (gray_level, gray_level, gray_level)
            cv2.drawContours(trace_img, [contour], -1, color, thickness=cv2.FILLED)
        
        overlay_frame = frame.copy()
        cv2.addWeighted(trace_img, 0.5, overlay_frame, 1, 0, overlay_frame)
        frames.append(cv2.cvtColor(overlay_frame, cv2.COLOR_BGR2RGB))
    
    cv2.imwrite(output_path, trace_img)
    path_overlay = first_frame.copy()
    cv2.addWeighted(trace_img, 0.5, path_overlay, 1, 0, path_overlay)
    cv2.imwrite(path_image_output, path_overlay)
    gif.release()
    
    # Save overlayed GIF using imageio
    imageio.mimsave(overlay_output, frames, format='GIF', duration=0.1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_gif", type=Path, help="Path to input GIF file")
    parser.add_argument("output_image", type=Path, help="Path to save the output trace image")
    parser.add_argument("overlay_output", type=Path, help="Path to save the overlayed GIF")
    parser.add_argument("path_image_output", type=Path, help="Path to save the traced path over the first frame")
    parser.add_argument("--max_dist", type=int, default=50, help="Max distance to track contours")
    parser.add_argument("--index", type=int, default=0, help="Index of the object to track")
    
    args = parser.parse_args()
    process_gif(str(args.input_gif), str(args.output_image), str(args.overlay_output), str(args.path_image_output), args.max_dist, args.index)

if __name__ == "__main__":
    main()
