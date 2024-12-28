import cv2
import numpy as np
import matplotlib.pyplot as plt


def compare_histograms_side_by_side(secret_image_path, extracted_image_path):
    """
    Membandingkan histogram antara gambar rahasia dan hasil ekstraksi secara side-by-side.
    
    Args:
        secret_image_path (str): Path ke gambar rahasia.
        extracted_image_path (str): Path ke gambar hasil ekstraksi.
    """
    # Load gambar
    secret_image = cv2.imread(secret_image_path, cv2.IMREAD_GRAYSCALE)
    extracted_image = cv2.imread(extracted_image_path, cv2.IMREAD_GRAYSCALE)
    
    if secret_image is None or extracted_image is None:
        raise ValueError("Salah satu gambar gagal dimuat. Periksa path gambar.")
    
    if secret_image.shape != extracted_image.shape:
        raise ValueError("Dimensi gambar tidak sama. Pastikan kedua gambar memiliki ukuran yang sama.")
    
    # Hitung histogram untuk setiap gambar
    hist_secret, _ = np.histogram(secret_image, bins=256, range=(0, 256))
    hist_extracted, _ = np.histogram(extracted_image, bins=256, range=(0, 256))
    
    # Normalisasi histogram
    hist_secret = hist_secret / hist_secret.sum()
    hist_extracted = hist_extracted / hist_extracted.sum()
    
    # Plot histogram side-by-side
    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram Secret Image
    axs[0].plot(hist_secret, color='blue')
    axs[0].set_title('Histogram: Secret Image')
    axs[0].set_xlabel('Pixel Intensity')
    axs[0].set_ylabel('Normalized Frequency')
    
    # Histogram Extracted Image
    axs[1].plot(hist_extracted, color='orange')
    axs[1].set_title('Histogram: Extracted Image')
    axs[1].set_xlabel('Pixel Intensity')
    axs[1].set_ylabel('Normalized Frequency')
    
    # Tampilkan plot
    plt.tight_layout()
    plt.show()


# Debug/Testing
if __name__ == '__main__':
    try:
        compare_histograms_side_by_side(
            'test/secret_image.png',   # Path ke gambar rahasia
            'test/extracted_image.png'  # Path ke gambar hasil ekstraksi
        )
    except ValueError as e:
        print(e)
