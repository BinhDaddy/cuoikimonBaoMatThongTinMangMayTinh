import cv2
import numpy as np
from watermark import DigitalWatermark
import matplotlib.pyplot as plt
import time
import os

def calculate_psnr(original, watermarked):
    """Tính PSNR giữa ảnh gốc và ảnh đã nhúng thủy vân"""
    mse = np.mean((original - watermarked) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(original, watermarked):
    """Tính SSIM giữa ảnh gốc và ảnh đã nhúng thủy vân"""
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    original = original.astype(np.float64)
    watermarked = watermarked.astype(np.float64)
    
    mu1 = cv2.GaussianBlur(original, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(watermarked, (11, 11), 1.5)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = cv2.GaussianBlur(original ** 2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(watermarked ** 2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(original * watermarked, (11, 11), 1.5) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return np.mean(ssim_map)

def main():
    # Tạo thư mục output nếu chưa tồn tại
    if not os.path.exists('output'):
        os.makedirs('output')
    
    # Đường dẫn ảnh
    host_image_path = 'input/lena.png'
    watermark_image_path = 'input/watermark.png'
    
    # Đọc ảnh gốc và ảnh thủy vân
    host_img = cv2.imread(host_image_path, cv2.IMREAD_GRAYSCALE)
    watermark_img = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Tính toán kích thước mới cho ảnh thủy vân (1/4 kích thước ảnh gốc)
    new_width = host_img.shape[1] // 4
    new_height = host_img.shape[0] // 4
    watermark_shape = (new_height, new_width)
    
    # Khởi tạo đối tượng DigitalWatermark
    watermark = DigitalWatermark()
    
    # Thực hiện nhúng thủy vân bằng LSB
    start_time = time.time()
    lsb_watermarked = watermark.lsb_embed_image(host_image_path, watermark_image_path, 'output/lsb_watermarked.png')
    lsb_time = time.time() - start_time
    
    # Thực hiện nhúng thủy vân bằng DCT
    start_time = time.time()
    dct_watermarked = watermark.dct_embed_image(host_image_path, watermark_image_path, 'output/dct_watermarked.png')
    dct_time = time.time() - start_time
    
    # Tính toán các chỉ số đánh giá cho LSB
    lsb_psnr = calculate_psnr(host_img, lsb_watermarked)
    lsb_ssim = calculate_ssim(host_img, lsb_watermarked)
    
    # Tính toán các chỉ số đánh giá cho DCT
    dct_psnr = calculate_psnr(host_img, dct_watermarked)
    dct_ssim = calculate_ssim(host_img, dct_watermarked)
    
    # Trích xuất ảnh thủy vân từ ảnh đã nhúng
    lsb_extracted = watermark.lsb_extract_image('output/lsb_watermarked.png', watermark_shape, 'output/lsb_extracted.png')
    dct_extracted = watermark.dct_extract_image('output/dct_watermarked.png', host_image_path, watermark_shape, 'output/dct_extracted.png')
    
    # Đánh giá độ bền vững với các loại tấn công
    attack_types = ['noise', 'blur', 'crop', 'rotate', 'compress']
    
    print("\nKết quả đánh giá chất lượng ảnh:")
    print("\nLSB:")
    print(f"Thời gian nhúng: {lsb_time:.2f} giây")
    print(f"PSNR: {lsb_psnr:.2f} dB")
    print(f"SSIM: {lsb_ssim:.4f}")
    
    print("\nDCT:")
    print(f"Thời gian nhúng: {dct_time:.2f} giây")
    print(f"PSNR: {dct_psnr:.2f} dB")
    print(f"SSIM: {dct_ssim:.4f}")
    
    # Đánh giá độ bền vững cho LSB
    print("\nĐánh giá độ bền vững của LSB:")
    lsb_robustness = watermark.evaluate_robustness(host_img, lsb_watermarked, watermark_img, attack_types)
    for attack_type, metrics in lsb_robustness.items():
        print(f"\n{attack_type.upper()}:")
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")
    
    # Đánh giá độ bền vững cho DCT
    print("\nĐánh giá độ bền vững của DCT:")
    dct_robustness = watermark.evaluate_robustness(host_img, dct_watermarked, watermark_img, attack_types)
    for attack_type, metrics in dct_robustness.items():
        print(f"\n{attack_type.upper()}:")
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")
    
    # Hiển thị ảnh gốc và ảnh thủy vân
    plt.figure(figsize=(15, 10))
    
    # Ảnh gốc và ảnh thủy vân
    plt.subplot(2, 3, 1)
    plt.imshow(host_img, cmap='gray')
    plt.title('Ảnh gốc')
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.imshow(watermark_img, cmap='gray')
    plt.title('Ảnh thủy vân')
    plt.axis('off')
    
    # Ảnh đã nhúng thủy vân LSB
    plt.subplot(2, 3, 3)
    plt.imshow(lsb_watermarked, cmap='gray')
    plt.title('Ảnh đã nhúng thủy vân (LSB)')
    plt.axis('off')
    
    # Ảnh đã nhúng thủy vân DCT
    plt.subplot(2, 3, 4)
    plt.imshow(dct_watermarked, cmap='gray')
    plt.title('Ảnh đã nhúng thủy vân (DCT)')
    plt.axis('off')
    
    # Ảnh thủy vân trích xuất từ LSB
    plt.subplot(2, 3, 5)
    plt.imshow(lsb_extracted, cmap='gray')
    plt.title('Ảnh thủy vân trích xuất (LSB)')
    plt.axis('off')
    
    # Ảnh thủy vân trích xuất từ DCT
    plt.subplot(2, 3, 6)
    plt.imshow(dct_extracted, cmap='gray')
    plt.title('Ảnh thủy vân trích xuất (DCT)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Hiển thị ảnh bị tấn công
    for attack_type in attack_types:
        plt.figure(figsize=(15, 10))
        
        # Ảnh gốc
        plt.subplot(2, 3, 1)
        plt.imshow(host_img, cmap='gray')
        plt.title('Ảnh gốc')
        plt.axis('off')
        
        # Ảnh thủy vân
        plt.subplot(2, 3, 2)
        plt.imshow(watermark_img, cmap='gray')
        plt.title('Ảnh thủy vân')
        plt.axis('off')
        
        # Ảnh LSB bị tấn công
        lsb_attacked = watermark.apply_attacks(lsb_watermarked, attack_type, {})
        plt.subplot(2, 3, 3)
        plt.imshow(lsb_attacked, cmap='gray')
        plt.title(f'Ảnh LSB bị tấn công ({attack_type})')
        plt.axis('off')
        
        # Ảnh DCT bị tấn công
        dct_attacked = watermark.apply_attacks(dct_watermarked, attack_type, {})
        plt.subplot(2, 3, 4)
        plt.imshow(dct_attacked, cmap='gray')
        plt.title(f'Ảnh DCT bị tấn công ({attack_type})')
        plt.axis('off')
        
        # Thủy vân trích xuất từ LSB bị tấn công
        lsb_extracted_attacked = watermark.lsb_extract_image(lsb_attacked, watermark_shape, f'output/{attack_type}_lsb_extracted.png')
        plt.subplot(2, 3, 5)
        plt.imshow(lsb_extracted_attacked, cmap='gray')
        plt.title(f'Thủy vân trích xuất từ LSB ({attack_type})')
        plt.axis('off')
        
        # Thủy vân trích xuất từ DCT bị tấn công
        dct_extracted_attacked = watermark.dct_extract_image(dct_attacked, host_image_path, watermark_shape, f'output/{attack_type}_dct_extracted.png')
        plt.subplot(2, 3, 6)
        plt.imshow(dct_extracted_attacked, cmap='gray')
        plt.title(f'Thủy vân trích xuất từ DCT ({attack_type})')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main() 