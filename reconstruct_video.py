import cv2
import os
import subprocess


def reconstruct_video(frames_folder, temp_video_path, fps):
    """
    Reconstruct video from frames.
    """
    frame_files = sorted(os.listdir(frames_folder))
    if not frame_files:
        raise ValueError("No frames found in the folder!")

    # Read sample frame to get dimensions
    sample_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width, layers = sample_frame.shape

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        out.write(frame)

    out.release()
    print(f"Video reconstructed and saved as {temp_video_path}")


def merge_audio(video_path, audio_path, output_video_path):
    """
    Merge audio into the reconstructed video.
    """
    command = [
        'ffmpeg',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-strict', 'experimental',
        '-shortest',
        output_video_path
    ]
    subprocess.run(command, check=True)
    print(f"Video with audio saved as {output_video_path}")


# Debug/Testing
if __name__ == '__main__':
    frames_folder = 'test/frames'
    temp_video_path = 'test/temp_video.avi'
    audio_path = 'test/audio/audio_output.mp3'
    final_video_path = 'test/output_video_WM.mp4'
    fps = 24

    # Step 1: Reconstruct Video from Frames
    reconstruct_video(frames_folder, temp_video_path, fps)

    # Step 2: Merge Original Audio with the Video
    merge_audio(temp_video_path, audio_path, final_video_path)
