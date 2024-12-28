import cv2
import numpy as np
import random


def extract_image_from_frame(frame_path, output_image_path, seed, image_shape):
    # Load frame
    frame = cv2.imread(frame_path)
    height, width, _ = frame.shape

    if image_shape[0] * image_shape[1] > height * width:
        raise ValueError(
            "Ukuran gambar rahasia melebihi kapasitas frame."
        )

    # Inisialisasi random seed dan koordinat acak
    random.seed(seed)
    coordinates = random.sample(
        [(x, y) for x in range(height) for y in range(width)],
        image_shape[0] * image_shape[1]
    )

    # Ekstrak channel biru dari frame
    blue_channel = frame[:, :, 0].flatten()

    # Ekstraksi bit LSB
    extracted_bits = [
        (blue_channel[x * width + y] & 1)
        for x, y in coordinates
    ]

    # Ubah bit menjadi array gambar
    extracted_bits = np.array(extracted_bits, dtype=np.uint8)
    extracted_image = extracted_bits.reshape(image_shape) * 255  # Skala ke 0-255

    cv2.imwrite(output_image_path, extracted_image)
    print(f"Secret image extracted and saved to {output_image_path}")


# Debug/Testing
if __name__ == '__main__':
    try:
        extract_image_from_frame(
            'test/frames_reconstructed/frame_0001.png',
            'test/extracted_image_reconstructed.png',
            seed=1234,
            image_shape=(100, 100)  # Sesuaikan dengan ukuran gambar rahasia
        )
    except ValueError as e:
        print(e)
