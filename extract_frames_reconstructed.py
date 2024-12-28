import cv2
import os
import subprocess

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_folder}/frame_{count:04d}.png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        count += 1

    cap.release()
    print(f"{count} frames extracted and saved in {output_folder}")


def extract_audio(video_path, audio_output_path):
    os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)
    command = [
        'ffmpeg',
        '-i', video_path,
        '-q:a', '0',
        '-map', 'a',
        audio_output_path
    ]
    subprocess.run(command, check=True)
    print(f"Audio extracted and saved as {audio_output_path}")


# Debug/Testing
if __name__ == '__main__':
    video_file = 'test/output_video_with_audio.mp4'
    frames_folder = 'test/frames_reconstructed'
    audio_file = 'test/audio/audio_output_reconstructed.mp3'
    
    # Extract frames
    extract_frames(video_file, frames_folder)
    
    # Extract audio
    extract_audio(video_file, audio_file)
