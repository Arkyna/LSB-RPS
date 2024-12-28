import cv2
import numpy as np
import random


def embed_image_to_frame(frame_path, image_path, seed):
    # Load frame dan gambar rahasia
    frame = cv2.imread(frame_path)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image_flat = image.flatten()
    
    height, width, _ = frame.shape
    frame_capacity = height * width  # Total bit yang bisa disisipkan

    if len(image_flat) > frame_capacity:
        raise ValueError(
            f"Gambar rahasia terlalu besar untuk disisipkan. "
            f"Kapasitas frame: {frame_capacity}, Dibutuhkan: {len(image_flat)}"
        )

    # Inisialisasi random seed dan koordinat acak
    random.seed(seed)
    coordinates = random.sample(
        [(x, y) for x in range(height) for y in range(width)],
        len(image_flat)
    )
    
    # Ekstrak channel biru dari frame
    blue_channel = frame[:, :, 0].flatten()

    # Lakukan embedding bitwise
    for (x, y), bit in zip(coordinates, image_flat):
        idx = x * width + y
        blue_channel[idx] = (blue_channel[idx] & 0xFE) | (bit & 1)
    
    # Kembalikan channel biru ke frame
    frame[:, :, 0] = blue_channel.reshape((height, width))
    cv2.imwrite(frame_path, frame)
    print(f"Image embedded into {frame_path}")


# Debug/Testing
if __name__ == '__main__':
    try:
        embed_image_to_frame(
            'test/frames/frame_0001.png',
            'test/secret_image.png',
            seed=1234
        )
    except ValueError as e:
        print(e)
