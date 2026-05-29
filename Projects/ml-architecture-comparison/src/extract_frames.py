import cv2
import os
import argparse


def extract_frames(video_path, output_dir, fps=2, max_duration=120):
    """
    Extract frames from a video file at a given frame rate.

    Args:
        video_path: path to the video file
        output_dir: folder to save extracted frames
        fps: how many frames to extract per second of video
        max_duration: maximum seconds to process (default: 120)
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: could not open video {video_path}")
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    # cap duration if video is too long
    effective_duration = min(duration, max_duration)
    max_frames_to_read = int(effective_duration * video_fps)

    print(f"  Video: {os.path.basename(video_path)}")
    print(f"  Duration: {duration:.1f}s | FPS: {video_fps:.1f} | Processing: {effective_duration:.1f}s")

    frame_interval = int(video_fps / fps)
    frame_number = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # stop if we've hit the max duration
        if frame_number >= max_frames_to_read:
            print(f"  Reached max duration of {max_duration}s, stopping")
            break

        if frame_number % frame_interval == 0:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            filename = f"{base_name}_frame{frame_number:04d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1

        frame_number += 1

    cap.release()
    print(f"  Saved {saved_count} frames")
    return saved_count


def process_class(class_name, base_dir="data/raw", fps=2, max_duration=120):
    """
    Extract frames from all videos for a given shot class.

    Args:
        class_name: name of the shot class e.g. 'forehand'
        base_dir: root of the raw data directory
        fps: frames per second to extract
    """
    video_dir = os.path.join(base_dir, "videos", class_name)
    image_dir = os.path.join(base_dir, "images", class_name)

    if not os.path.exists(video_dir):
        print(f"Error: video directory not found: {video_dir}")
        return

    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".webm")
    videos = [f for f in os.listdir(video_dir) if f.lower().endswith(video_extensions)]

    if not videos:
        print(f"No videos found in {video_dir}")
        return

    print(f"\nProcessing class '{class_name}' — {len(videos)} video(s) found")
    total = 0

    for video_file in videos:
        video_path = os.path.join(video_dir, video_file)
        total += extract_frames(video_path, image_dir, fps=fps, max_duration=max_duration)

    print(f"Total frames extracted for '{class_name}': {total}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from tennis shot videos")
    parser.add_argument("class_name", help="Shot class name e.g. forehand, backhand, serve")
    parser.add_argument("--base_dir", default="data/raw", help="Base data directory (default: data/raw)")
    parser.add_argument("--fps", type=int, default=2, help="Frames per second to extract (default: 2)")
    parser.add_argument("--max_duration", type=int, default=120, help="Max seconds to process per video (default: 120)")
    args = parser.parse_args()

    process_class(args.class_name, base_dir=args.base_dir, fps=args.fps, max_duration=args.max_duration)