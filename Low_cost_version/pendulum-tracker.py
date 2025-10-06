# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Copyright (c) Jacob Berman and Michael Berman
# Adapted for interactive use on memory constrained systems and laptops.
# Specific modifications for Apple Silicon MBP.
import torch
import argparse
import numpy as np
import cv2  # Using OpenCV for GUI
import os
import imageio.v3 as iio
import imageio
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import pandas as pd
from cotracker.utils.visualizer import Visualizer

import sys

# Note: This demo uses the online model for memory efficiency.

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# --- Check for MPS and re-execute if necessary ---
if DEFAULT_DEVICE == "mps":
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "1":
        print("Re-executing with PYTORCH_ENABLE_MPS_FALLBACK=1...")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

# Global list to store clicked points
points_to_track = []
# A copy of the frame to draw on
frame_to_draw = None
global point_radius


def select_points_callback(event, x, y, flags, param):
    """Mouse callback function to capture points."""
    global points_to_track, frame_to_draw, point_radius
    if event == cv2.EVENT_LBUTTONDOWN:
        point_index = len(points_to_track)
        points_to_track.append((x, y))
        # Draw a circle on the frame for visual feedback
        cv2.circle(frame_to_draw, (x, y), point_radius, (0, 255, 0), -1)
        # Draw the point index next to the circle for clarity
        cv2.putText(
            frame_to_draw,
            str(point_index),
            (x + point_radius + 2, y + point_radius),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255), # Red color for visibility
            2,
        )


def precompute_colors(tracks, query_frames):
    """
    Calculates track colors based on their starting position.
    `tracks` is a numpy array of shape (T, N, 2).
    `query_frames` is an int or a tensor of start frames.
    """
    T, N, _ = tracks.shape
    vector_colors = np.zeros((T, N, 3))
    color_map = plt.get_cmap("gist_rainbow")

    if isinstance(query_frames, torch.Tensor):
        query_frames_np = query_frames.long().cpu().numpy()
        point_indices = np.arange(N)
        initial_y_coords = tracks[query_frames_np, point_indices, 1]
        y_min, y_max = initial_y_coords.min(), initial_y_coords.max()
    else:  # integer
        y_min, y_max = (
            tracks[query_frames, :, 1].min(),
            tracks[query_frames, :, 1].max(),
        )

    norm = plt.Normalize(y_min, y_max)
    for n in range(N):
        if isinstance(query_frames, torch.Tensor):
            query_frame_ = query_frames[n].long().item()
        else:
            query_frame_ = query_frames
        color = color_map(norm(tracks[query_frame_, n, 1]))
        color = np.array(color[:3])[None] * 255
        vector_colors[:, n] = np.repeat(color, T, axis=0)
    return vector_colors


def get_points_from_user(first_frame):
    """
    Displays the first frame of the video and lets the user select points with the mouse.
    Returns a list of (x, y) coordinates.
    """
    global points_to_track, frame_to_draw
    points_to_track = []  # Reset points
    frame_to_draw = first_frame.copy()

    window_name = "Select Points to Track (ENTER: start, u: undo, c: clear, q: quit)"
    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, select_points_callback)

    print("\n--- Interactive Point Selection ---")
    print("Click on the image to select points to track.")
    print("Use hjkl keys to adjust the last point by one pixel.")
    print("  - Press 'ENTER' to confirm your selection and start tracking.")
    print("  - Press 'u' to undo the last point.")
    print("  - Press 'c' to clear all selected points.")
    print("  - Press 'q' to quit without tracking.")

    while True:
        cv2.imshow(window_name, frame_to_draw)
        key = cv2.waitKey(20)

        if key != -1 and points_to_track:
            x, y = points_to_track[-1]
            moved = False
            # Use "vim" keys
            if key == ord('h'):  # Left
                x -= 1
                moved = True
            elif key == ord('k'):  # Up
                y -= 1
                moved = True
            elif key == ord('l'):  # Right
                x += 1
                moved = True
            elif key == ord('j'):  # Down
                y += 1
                moved = True

            if moved:
                points_to_track[-1] = (x, y)
                # Redraw all points
                frame_to_draw = first_frame.copy()
                for i, (px, py) in enumerate(points_to_track):
                    cv2.circle(frame_to_draw, (px, py), point_radius, (0, 255, 0), -1)
                    cv2.putText(
                        frame_to_draw,
                        str(i),
                        (px + point_radius + 2, py + point_radius),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
                continue

        key = key & 0xFF
        if key == 13:  # Enter key
            break
        elif key == ord('u'):
            if points_to_track:
                points_to_track.pop()
                frame_to_draw = first_frame.copy()
                # Redraw all points with their indices
                for i, (px, py) in enumerate(points_to_track):
                    cv2.circle(frame_to_draw, (px, py), point_radius, (0, 255, 0), -1)
                    cv2.putText(
                        frame_to_draw,
                        str(i),
                        (px + point_radius + 2, py + point_radius),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
                print("Last point removed.")
            else:
                print("No points to remove.")
        elif key == ord('c'):  # 'c' to clear
            points_to_track = []
            frame_to_draw = first_frame.copy()
            print("Points cleared.")
        elif key == ord('q'):  # 'q' to quit
            points_to_track = []
            break

    cv2.destroyAllWindows()
    return points_to_track


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively select points to track in a video.")
    parser.add_argument(
        "--video_path", default="./assets/apple.mp4", help="Path to a video file."
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Path to CoTracker model parameters."
    )
    parser.add_argument(
        "--backward_tracking", action="store_true", help="Compute tracks in both directions."
    )
    parser.add_argument(
        "--export_tracks", type=str, default=None,
        help="Path to save the predicted tracks as a CSV file. e.g., ./saved_tracks.csv"
    )
    parser.add_argument(
        "--point_radius", type=int, default=3,
        help="Radius of the circle used to visualize selected points."
    )
    args = parser.parse_args()

    # --- 1. Get points from user ---
    try:
        # Use imageio to read the first frame (as RGB)
        first_frame_rgb = iio.imread(args.video_path, index=0, plugin="FFMPEG")
        # Convert to BGR for display with OpenCV
        first_frame_bgr = cv2.cvtColor(first_frame_rgb, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Error: {e}")
        exit()

    global point_radius
    point_radius = args.point_radius

    selected_points = get_points_from_user(first_frame_bgr)

    if not selected_points:
        print("No points selected. Exiting.")
        exit()

    print(f"\nTracking {len(selected_points)} points: {selected_points}")

    # --- 2. Load online model ---
    print("Loading CoTracker online model...")
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")
    model = model.to(DEFAULT_DEVICE)

    # --- 3. Prepare queries and tracking loop ---
    queries = torch.tensor([[0, x, y] for x, y in selected_points]).float()[None].to(DEFAULT_DEVICE)

    window_frames = []
    buffer_size = 2 * model.step

    def _process_step(window_frames, is_first_step, queries=None):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )
        if is_first_step:
            return model(video_chunk, is_first_step=True, queries=queries)
        else:
            return model(video_chunk)

    start_time = time.time()
    is_first_step = True
    i = 0
    print("Running online tracking...")
    video_iterator = iio.imiter(args.video_path, plugin="FFMPEG")
    for i, frame in enumerate(tqdm(video_iterator, unit=" frames", desc="Tracking")):
        # imageio reads in RGB, which is what the model expects.
        window_frames.append(frame)

        if len(window_frames) > buffer_size:
            window_frames.pop(0)

        if i % model.step == 0 and i > 0:
            pred_tracks, pred_visibility = _process_step(
                window_frames, is_first_step, queries=queries
            )
            is_first_step = False

    # Process the final frames
    pred_tracks, pred_visibility = _process_step(
        window_frames, is_first_step, queries=queries
    )

    print(f"Tracking complete. {i+1} frames processed in {time.time() - start_time:.2f} seconds.")

    # --- 4. Export track data if requested ---
    if args.export_tracks:
        print(f"Exporting tracks to {args.export_tracks}...")
        # Ensure the directory exists
        export_dir = os.path.dirname(args.export_tracks)
        if export_dir:
            os.makedirs(export_dir, exist_ok=True)

        # Squeeze batch dimension and move to CPU
        tracks_np = pred_tracks[0].cpu().numpy()  # Shape: (T, N, 2)
        visibility_np = pred_visibility[0].cpu().numpy()  # Shape: (T, N)

        num_frames, num_points, _ = tracks_np.shape

        data_to_save = []
        for frame_idx in range(num_frames):
            # Each row in the CSV will represent one frame
            row_data = {"frame_index": frame_idx}
            for point_id in range(num_points):
                # Add data for each point as separate columns
                row_data[f"point_{point_id}_x"] = tracks_np[frame_idx, point_id, 0]
                row_data[f"point_{point_id}_y"] = tracks_np[frame_idx, point_id, 1]
            data_to_save.append(row_data)
        pd.DataFrame(data_to_save).to_csv(args.export_tracks, index=False)
        print("Export complete.")

    # --- 5. Visualize results memory-efficiently ---
    print("Visualizing tracks...")

    query_frame_to_vis = queries[0, :, 0].cpu()
    all_track_colors = precompute_colors(
        pred_tracks[0].cpu().numpy(), query_frame_to_vis
    )

    seq_name = os.path.basename(args.video_path)
    vis = Visualizer(save_dir="./saved_videos", pad_value=120, linewidth=3)
    output_video_path = os.path.join(vis.save_dir, f"interactive_{seq_name}")
    os.makedirs(vis.save_dir, exist_ok=True)

    with iio.imopen(args.video_path, "r", plugin="FFMPEG") as reader:
        fps = reader.metadata().get("fps", 30)

    start_time = time.time()
    i = 0

    video_iterator_vis = iio.imiter(args.video_path, plugin="FFMPEG")
    with imageio.get_writer(output_video_path, fps=fps) as writer:
        for i, frame in enumerate(tqdm(video_iterator_vis, total=pred_tracks.shape[1], desc="Visualizing", unit=" frames")):
            if i >= pred_tracks.shape[1]:
                break

            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1)[None, None].float().to(DEFAULT_DEVICE)
            tracks_for_frame = pred_tracks[:, i:i+1]
            visibility_for_frame = pred_visibility[:, i:i+1]

            frame_with_tracks = vis.draw_tracks_on_video(
                frame_tensor,
                tracks_for_frame,
                visibility_for_frame,
                query_frame=0,
                vector_colors=all_track_colors[i : i + 1],
            )
            output_frame = (frame_with_tracks[0, 0].permute(1, 2, 0).cpu().numpy()).astype(np.uint8)
            writer.append_data(output_frame)

    print(f"Visualization complete. {i+1} frames visualized in {time.time() - start_time:.2f} seconds.")
    print(f"Video saved to: {output_video_path}")
