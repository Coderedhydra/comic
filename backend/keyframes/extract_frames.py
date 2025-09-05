import cv2
import os
import subprocess

def _extract_with_ffmpeg(input_video, output_path, start_time, end_time, frame_rate):
    """Fallback extraction using ffmpeg when OpenCV fails (handles codecs like AV1)."""
    os.makedirs(output_path, exist_ok=True)
    # Build ffmpeg command
    # Use -loglevel error to suppress noise, -y to overwrite
    duration = end_time - start_time
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', input_video,
        '-t', str(duration),
        '-vf', f'fps={frame_rate}',
        os.path.join(output_path, 'frame_%03d.png'),
        '-loglevel', 'error'
    ]
    try:
        subprocess.run(cmd, check=True)
        # Return list of generated frames
        frames = sorted([os.path.join(output_path, f) for f in os.listdir(output_path) if f.startswith('frame_')])
        return frames
    except Exception as e:
        print(f"FFmpeg extraction failed: {e}")
        return []

def extract_frames(input_video, output_path, start_time, end_time, frame_rate):
    cap = cv2.VideoCapture(input_video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or not cap.isOpened():
        # OpenCV failed to open – try ffmpeg fallback
        cap.release()
        return _extract_with_ffmpeg(input_video, output_path, start_time, end_time, frame_rate)

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame

    frame_count = 0
    frames = []
    while current_frame < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if current_frame % max(int(fps / frame_rate), 1) == 0:
            # Save the frame
            os.makedirs(output_path, exist_ok=True)
            frame_filename = f"frame_{frame_count}.png"
            frame_path = os.path.join(output_path, frame_filename)
            frames.append(frame_path)
            cv2.imwrite(frame_path, frame)
            frame_count += 1

        current_frame += 1

    # Fallback: if no frames were extracted, attempt ffmpeg extraction
    if not frames:
        cap.release()
        return _extract_with_ffmpeg(input_video, output_path, start_time, end_time, frame_rate)

    cap.release()
    return frames