# python video_cutter.py "input\videos\input_name.mp4" "input\template\sidebar_template.png" --output "output\folder"

import cv2
import numpy as np
import argparse
import time
import os
import json
from datetime import datetime
from typing import Dict, Any, List

def analyze_frame(frame: np.ndarray, analysis_objects: Dict[str, Any]) -> Dict[str, Any]:
    """
    Analyzes a single frame using ORB feature matching to find a template.
    """
    orb = analysis_objects['orb']
    bf_matcher = analysis_objects['bf_matcher']
    template_kp = analysis_objects['template_kp']
    template_des = analysis_objects['template_des']
    template_shape = analysis_objects['template_img'].shape[:2]

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_kp, frame_des = orb.detectAndCompute(gray_frame, None)

    if template_des is None or frame_des is None:
        return {"match_count": 0, "box_points": None}

    matches = bf_matcher.knnMatch(template_des, frame_des, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    
    box_points = None
    if len(good_matches) > analysis_objects['config']['good_match_threshold']:
        src_pts = np.float32([template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_kp[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        homography_matrix, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if homography_matrix is not None:
            h, w = template_shape
            pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, homography_matrix)
            box_points = np.int32(dst)

    return {"match_count": len(good_matches), "box_points": box_points}

def annotate_frame(frame: np.ndarray, analysis_results: Dict[str, Any], fps: float) -> np.ndarray:
    """
    Draws debug information onto a frame.
    """
    annotated_frame = frame.copy()
    match_count = analysis_results['match_count']
    box_points = analysis_results['box_points']

    match_text = f"Good Matches: {match_count}"
    cv2.putText(annotated_frame, match_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    frame_width = frame.shape[1]
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(annotated_frame, fps_text, (frame_width - 250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    if box_points is not None:
        cv2.polylines(annotated_frame, [box_points], True, (0, 255, 0), 3, cv2.LINE_AA)

    return annotated_frame


def process_video(video_path: str, analysis_objects: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Processes a single video file based on the provided configuration.

    Args:
        video_path (str): The full path to the input video file.
        analysis_objects (Dict[str, Any]): Pre-initialized objects for feature matching.
        config (Dict[str, Any]): The global configuration dictionary.

    Returns:
        Dict[str, Any]: A dictionary containing statistics about the processed video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Could not open video file: {video_path}"}

    # Prepare output path
    base_name = os.path.basename(video_path)
    file_name_without_ext = os.path.splitext(base_name)[0]
    output_filename = f"out_{file_name_without_ext}.mp4"
    full_output_path = os.path.join(config['output_folder'], output_filename)

    # Video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    output_writer = cv2.VideoWriter(full_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    frames_processed = 0
    frames_removed = 0
    
    # FPS counter variables
    fps_counter, fps_start_time, processing_fps = 0, time.time(), 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frames_processed += 1
        print(f"  -> Processing frame {frames_processed}/{total_frames}...", end='\r')

        analysis_results = analyze_frame(frame, analysis_objects)
        match_count = analysis_results['match_count']
        
        panel_detected = match_count >= config['good_match_threshold']

        if config['debug']:
            fps_counter += 1
            elapsed = time.time() - fps_start_time
            if elapsed > 1.0:
                processing_fps = fps_counter / elapsed
                fps_counter, fps_start_time = 0, time.time()

            debug_frame = annotate_frame(frame, analysis_results, processing_fps)
            cv2.imshow('Debug Preview', debug_frame)
            output_writer.write(debug_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\nProcessing interrupted by user.")
                break
        
        if panel_detected:
            frames_removed += 1
            if not config['debug']: # In debug mode, we write all frames
                continue

        if not config['debug']:
             output_writer.write(frame)

    cap.release()
    output_writer.release()
    print() # Newline after progress indicator
    
    return {
        "total_frames": total_frames,
        "frames_removed": frames_removed,
        "output_path": full_output_path,
        "error": None
    }

def main():
    """
    Main entry point. Loads config, orchestrates batch processing, and logs results.
    """
    parser = argparse.ArgumentParser(description="Batch process videos to remove frames with a specific panel.")
    parser.add_argument("config_file", help="Path to the JSON configuration file.")
    args = parser.parse_args()

    try:
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at '{args.config_file}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not parse '{args.config_file}'. Please check for JSON syntax errors.")
        return

    # --- Initialize shared resources ---
    os.makedirs(config['output_folder'], exist_ok=True)
    log_path = os.path.join(config['output_folder'], "processing_report.log")
    
    template_img = cv2.imread(config['template_input'])
    if template_img is None:
        print(f"Error: Could not load template image from '{config['template_input']}'")
        return

    analysis_objects = {
        'orb': cv2.ORB_create(nfeatures=1000),
        'bf_matcher': cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
        'template_img': template_img,
        'config': config
    }
    kp, des = analysis_objects['orb'].detectAndCompute(template_img, None)
    analysis_objects['template_kp'], analysis_objects['template_des'] = kp, des

    if config['debug']:
        cv2.namedWindow('Debug Preview', cv2.WINDOW_NORMAL)

    # --- Main Batch Loop ---
    with open(log_path, 'w') as log_file:
        log_file.write(f"--- Batch Processing Report ---\n")
        log_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for i, video_file in enumerate(config['input_files']):
            print(f"\nProcessing file {i+1}/{len(config['input_files'])}: {video_file}")
            start_time = datetime.now()
            
            stats = process_video(video_file, analysis_objects, config)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Log results
            log_file.write(f"--- File: {os.path.basename(video_file)} ---\n")
            log_file.write(f"Start Time:       {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"End Time:         {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"Processing Time:  {duration:.2f} seconds\n")
            if stats['error']:
                log_file.write(f"Status:           ERROR - {stats['error']}\n\n")
                print(f"Error processing file: {stats['error']}")
            else:
                log_file.write(f"Input Frames:     {stats['total_frames']}\n")
                log_file.write(f"Frames Removed:   {stats['frames_removed']}\n")
                log_file.write(f"Output Path:      {stats['output_path']}\n\n")
                print(f"Finished. Removed {stats['frames_removed']} frames. Took {duration:.2f}s.")

    if config['debug']:
        cv2.destroyAllWindows()
    
    print(f"\nBatch processing complete. Report saved to '{log_path}'")

if __name__ == '__main__':
    main()
