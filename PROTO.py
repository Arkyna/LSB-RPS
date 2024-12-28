import cv2
import os
import subprocess
import numpy as np
import random

def extract_frames(video_path, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(f"{output_folder}/frame_{count:04d}_embed.png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        count += 1

    cap.release()
    print(f"{count} frames extracted and saved in {output_folder}")

def embed_image_to_frame(frame_path, image_path, seed):
    frame = cv2.imread(frame_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_flat = image.flatten()
    
    height, width, _ = frame.shape
    frame_capacity = height * width

    if len(image_flat) > frame_capacity:
        raise ValueError(
            f"Gambar rahasia terlalu besar untuk disisipkan. "
            f"Kapasitas frame: {frame_capacity}, Dibutuhkan: {len(image_flat)}"
        )

    random.seed(seed)
    coordinates = random.sample(
        [(x, y) for x in range(height) for y in range(width)],
        len(image_flat)
    )
    
    blue_channel = frame[:, :, 0].flatten()
    for (x, y), bit in zip(coordinates, image_flat):
        idx = x * width + y
        blue_channel[idx] = (blue_channel[idx] & 0xFE) | (bit & 1)
    
    frame[:, :, 0] = blue_channel.reshape((height, width))
    cv2.imwrite(frame_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    print(f"Image embedded into {frame_path}")

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

def reconstruct_video(frames_folder, temp_video_path, fps):
    frame_files = sorted(os.listdir(frames_folder))
    if not frame_files:
        raise ValueError("No frames found in the folder!")

    sample_frame = cv2.imread(os.path.join(frames_folder, frame_files[0]))
    height, width, layers = sample_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'FFV1')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue
        out.write(frame)

    out.release()
    print(f"Lossless video reconstructed and saved as {temp_video_path}")

def merge_audio(video_path, audio_path, output_video_path):
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

def confirm_action():
    while True:
        confirm = input("Are you sure you want to proceed with embedding and reconstructing the video? (yes/no): ")
        if confirm.lower() == 'yes':
            return True
        elif confirm.lower() == 'no':
            print("Operation cancelled.")
            return False
        else:
            print("Invalid input. Please type 'yes' or 'no'.")

if __name__ == '__main__':
    while True:
        print("\n=== Watermark Video Tool ===")
        print("1. Embed watermark and reconstruct video")
        print("2. Exit")
        choice = input("Choose an option: ")

        if choice == '1':
            if confirm_action():
                video_file = 'test/input_video.mp4'
                frames_folder = 'test/frames_embedded'
                audio_file = 'test/audio/audio_output.mp3'
                temp_video_path = 'test/temp_video.avi'
                final_video_path = 'test/output_video_with_audio.mp4'
                secret_image_path = 'test/secret_image.png'
                fps = 24

                extract_frames(video_file, frames_folder)
                extract_audio(video_file, audio_file)
                embed_image_to_frame(
                    f'{frames_folder}/frame_0001.png',
                    secret_image_path,
                    seed=1234
                )
                reconstruct_video(frames_folder, temp_video_path, fps)
                merge_audio(temp_video_path, audio_file, final_video_path)
                print("Watermark embedding and reconstruction completed!")

        elif choice == '2':
            print("Exiting... Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")
