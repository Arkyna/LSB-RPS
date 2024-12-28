import cv2
import numpy as np
import math
from skimage.metrics import structural_similarity as ssim


def calculate_statistics(image1_path, image2_path):
    """
    Menghitung MSE, PSNR, dan SSIM antara dua gambar.
    
    Args:
        image1_path (str): Path ke gambar pertama.
        image2_path (str): Path ke gambar kedua.
    """
    # Load gambar dalam grayscale
    image1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
    
    if image1 is None or image2 is None:
        raise ValueError("Salah satu gambar gagal dimuat. Periksa path gambar.")
    
    if image1.shape != image2.shape:
        raise ValueError("Dimensi gambar tidak sama. Pastikan kedua gambar memiliki ukuran yang sama.")
    
    # Hitung MSE
    mse = np.mean((image1 - image2) ** 2)
    
    # Hitung PSNR
    if mse == 0:
        psnr = float('inf')  # Jika gambar identik, PSNR tak terhingga
    else:
        PIXEL_MAX = 255.0
        psnr = 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    
    # Hitung SSIM
    ssim_index = ssim(image1, image2, data_range=PIXEL_MAX)
    
    # Cetak hasil
    print(f"Perbandingan antara '{image1_path}' dan '{image2_path}':")
    print(f"- MSE: {mse:.4f}")
    print(f"- PSNR: {psnr:.4f} dB")
    print(f"- SSIM: {ssim_index:.4f}")


# Path Gambar Langsung di Kode
if __name__ == '__main__':
    image1_path = 'essentials/image1.png'  # Ganti path kalau perlu
    image2_path = 'essentials/image2.png'  # Ganti path kalau perlu
    
    try:
        calculate_statistics(image1_path, image2_path)
    except ValueError as e:
        print(e)
