import cv2
import numpy as np
import argparse
import time
import os
import json
import multiprocessing
import sys
from typing import Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- 1. Worker Function ---

def process_video_chunk(chunk_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Worker: Reads a slice of the video, processes it, updates shared progress.
    """
    video_path = chunk_args['video_path']
    start_frame = chunk_args['start_frame']
    end_frame = chunk_args['end_frame']
    chunk_index = chunk_args['chunk_index']
    config = chunk_args['config']
    progress_dict = chunk_args['progress_dict'] # Shared dictionary
    
    # --- Initialization ---
    template_img = cv2.imread(config['template_input'])
    if template_img is None:
        return {"error": f"Template not found: {config['template_input']}"}

    orb = cv2.ORB_create(nfeatures=1000)
    template_kp, template_des = orb.detectAndCompute(template_img, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    temp_filename = f"temp_{os.path.basename(video_path)}_chunk_{chunk_index}.mp4"
    temp_output_path = os.path.join(config['output_folder'], temp_filename)
    
    writer = cv2.VideoWriter(temp_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    current_frame = start_frame
    frames_written = 0
    total_chunk_frames = end_frame - start_frame
    
    scale = config.get('processing_scale', 0.5)
    threshold = config['good_match_threshold']

    # --- Processing Loop ---
    try:
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret: break

            # Detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if scale != 1.0:
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            
            _, frame_des = orb.detectAndCompute(gray, None)
            
            panel_detected = False
            if template_des is not None and frame_des is not None:
                matches = bf.knnMatch(template_des, frame_des, k=2)
                good_count = sum(1 for m, n in matches if m.distance < 0.75 * n.distance)
                if good_count > threshold:
                    panel_detected = True

            # Write
            if not panel_detected:
                writer.write(frame)
                frames_written += 1
            
            current_frame += 1
            
            # UPDATE PROGRESS (Every 10 frames to reduce overhead)
            if current_frame % 10 == 0:
                progress = (current_frame - start_frame) / total_chunk_frames
                progress_dict[chunk_index] = progress

    except Exception as e:
        return {"error": str(e)}
    finally:
        # Ensure progress hits 100% on finish
        progress_dict[chunk_index] = 1.0
        cap.release()
        writer.release()
    
    return {
        "temp_path": temp_output_path, 
        "frames_written": frames_written, 
        "error": None
    }


# --- 2. Orchestrator ---

def process_single_video_multicore(video_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    print(f"--> Analyzing: {os.path.basename(video_path)}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Could not open {video_path}"}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # Setup Workers
    num_cores = multiprocessing.cpu_count()
    workers = min(num_cores, max(1, total_frames // 200)) # Ensure at least 200 frames per worker
    frames_per_chunk = total_frames // workers
    
    # Shared Progress Manager
    manager = multiprocessing.Manager()
    progress_dict = manager.dict()
    
    chunks = []
    for i in range(workers):
        start = i * frames_per_chunk
        end = (i + 1) * frames_per_chunk if i < workers - 1 else total_frames
        progress_dict[i] = 0.0 # Initialize progress
        
        chunks.append({
            'video_path': video_path,
            'start_frame': start,
            'end_frame': end,
            'chunk_index': i,
            'config': config,
            'progress_dict': progress_dict
        })

    print(f"    Splitting into {workers} chunks on {workers} cores.")
    
    # Execute and Monitor
    temp_files = [None] * workers
    total_frames_kept = 0
    errors = []

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_video_chunk, chunk): chunk['chunk_index'] for chunk in chunks}
        
        # MONITOR LOOP
        # While tasks are running, we print the status
        while True:
            # Check if all finished
            done_count = sum(1 for f in futures if f.done())
            
            # Construct Status String
            status_str = "\r    Progress: "
            for i in range(workers):
                p = progress_dict.get(i, 0.0)
                status_str += f"[C{i}: {int(p*100):02d}%] "
            
            sys.stdout.write(status_str)
            sys.stdout.flush()
            
            if done_count == len(futures):
                break
            
            time.sleep(0.2) # Update 5 times a second
        
        print() # New line after progress bar

        # Gather results
        for future in as_completed(futures):
            idx = futures[future]
            res = future.result()
            if res.get('error'):
                errors.append(res['error'])
            else:
                temp_files[idx] = res['temp_path'] # Store in correct order
                total_frames_kept += res['frames_written']

    if errors:
        return {"error": f"Worker errors: {', '.join(errors)}"}

    # Merge
    print(f"    Merging temp files...")
    output_filename = f"processed_{os.path.basename(video_path)}"
    final_output_path = os.path.join(config['output_folder'], output_filename)
    
    final_writer = cv2.VideoWriter(final_output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    for temp_file in temp_files:
        if temp_file and os.path.exists(temp_file):
            cap_temp = cv2.VideoCapture(temp_file)
            while True:
                ret, frame = cap_temp.read()
                if not ret: break
                final_writer.write(frame)
            cap_temp.release()
            os.remove(temp_file)

    final_writer.release()
    
    return {
        "path": final_output_path,
        "frames_removed": total_frames - total_frames_kept,
        "error": None
    }


# --- 3. Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to config file.")
    parser.add_argument("--scale", type=float, default=0.5)
    args = parser.parse_args()

    # Simplified Config Load
    config = {
        "template_input": "input/template/sidebar_template.png",
        "output_folder": "output/processed_videos",
        "good_match_threshold": 100,
        "processing_scale": args.scale,
        "input_files": []
    }

    if args.config:
        with open(args.config, 'r') as f:
            config.update(json.load(f))
    else:
        # Auto-detect logic
        input_dir = "input/videos"
        if os.path.exists(input_dir):
            config['input_files'] = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.mp4', '.mkv'))]

    os.makedirs(config['output_folder'], exist_ok=True)

    for i, video_file in enumerate(config['input_files']):
        print(f"\n[{i+1}/{len(config['input_files'])}] File: {video_file}")
        start = time.time()
        res = process_single_video_multicore(video_file, config)
        if res['error']:
            print(f"ERROR: {res['error']}")
        else:
            print(f"DONE. Removed {res['frames_removed']} frames. Took {time.time() - start:.2f}s")

if __name__ == '__main__':
    main()