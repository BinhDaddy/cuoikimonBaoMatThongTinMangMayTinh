import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

class DigitalWatermark:
    def __init__(self):
        pass

    def text_to_binary(self, text):
        """Chuyển đổi văn bản hoặc bytes thành chuỗi nhị phân"""
        if isinstance(text, bytes):
            # Nếu là bytes, chuyển từng byte thành nhị phân
            binary = ''.join(format(byte, '08b') for byte in text)
        else:
            # Nếu là string, chuyển từng ký tự thành nhị phân
        binary = ''.join(format(ord(char), '08b') for char in text)
        return binary

    def binary_to_text(self, binary):
        """Chuyển đổi chuỗi nhị phân thành văn bản"""
        text = ''
        for i in range(0, len(binary), 8):
            byte = binary[i:i+8]
            text += chr(int(byte, 2))
        return text

    def image_to_binary(self, image):
        """Chuyển đổi ảnh thành chuỗi nhị phân"""
        # Chuyển ảnh thành mảng numpy
        img_array = np.array(image)
        # Chuyển mảng thành chuỗi nhị phân
        binary = ''.join(format(pixel, '08b') for pixel in img_array.flatten())
        return binary

    def binary_to_image(self, binary, shape):
        """Chuyển đổi chuỗi nhị phân thành ảnh"""
        # Chuyển chuỗi nhị phân thành mảng numpy
        img_array = np.array([int(binary[i:i+8], 2) for i in range(0, len(binary), 8)])
        # Reshape mảng thành ảnh
        img_array = img_array.reshape(shape)
        return img_array

    def lsb_embed_image(self, host_image_path, watermark_image_path, output_path):
        """Nhúng ảnh thủy vân vào ảnh gốc sử dụng thuật toán LSB"""
        # Đọc ảnh gốc và ảnh thủy vân
        host_img = cv2.imread(host_image_path, cv2.IMREAD_GRAYSCALE)
        watermark_img = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize ảnh thủy vân để phù hợp với ảnh gốc
        watermark_img = cv2.resize(watermark_img, (host_img.shape[1] // 4, host_img.shape[0] // 4))
        
        # Chuyển đổi ảnh thủy vân thành nhị phân
        watermark_binary = self.image_to_binary(watermark_img)
        
        # Kiểm tra kích thước ảnh gốc có đủ để nhúng ảnh thủy vân không
        if len(watermark_binary) > host_img.size:
            raise ValueError("Ảnh thủy vân quá lớn so với ảnh gốc")
        
        # Nhúng ảnh thủy vân vào ảnh gốc
        flat_host = host_img.flatten()
        for i in range(len(watermark_binary)):
            flat_host[i] = (flat_host[i] & 254) | int(watermark_binary[i])
        
        # Tạo ảnh mới từ mảng đã nhúng thủy vân
        watermarked_img = flat_host.reshape(host_img.shape)
        
        # Lưu ảnh đã nhúng thủy vân
        if not output_path.lower().endswith('.png'):
            output_path = output_path.rsplit('.', 1)[0] + '.png'
        cv2.imwrite(output_path, watermarked_img)
        return watermarked_img

    def lsb_extract_image(self, watermarked_image, watermark_shape, output_path):
        """Trích xuất ảnh thủy vân từ ảnh đã nhúng sử dụng thuật toán LSB"""
        # Kiểm tra nếu watermarked_image là đường dẫn file
        if isinstance(watermarked_image, str):
            watermarked_img = cv2.imread(watermarked_image, cv2.IMREAD_GRAYSCALE)
        else:
            watermarked_img = watermarked_image
        
        # Đảm bảo ảnh là kiểu uint8
        watermarked_img = watermarked_img.astype(np.uint8)
        
        # Trích xuất ảnh thủy vân
        flat_watermarked = watermarked_img.flatten()
        binary_message = ''
        
        # Số bit cần trích xuất = số pixel của ảnh thủy vân * 8
        num_bits = watermark_shape[0] * watermark_shape[1] * 8
        
        # Đảm bảo số bit không vượt quá kích thước ảnh
        num_bits = min(num_bits, len(flat_watermarked))
        
        for i in range(num_bits):
            binary_message += str(flat_watermarked[i] & 1)
        
        # Chuyển đổi nhị phân thành ảnh
        extracted_img = self.binary_to_image(binary_message, watermark_shape)
        
        # Lưu ảnh đã trích xuất
        if not output_path.lower().endswith('.png'):
            output_path = output_path.rsplit('.', 1)[0] + '.png'
        cv2.imwrite(output_path, extracted_img)
        return extracted_img

    def dct_embed_image(self, host_image_path, watermark_image_path, output_path):
        """Nhúng ảnh thủy vân vào ảnh gốc sử dụng biến đổi DCT"""
        # Đọc ảnh gốc và ảnh thủy vân
        host_img = cv2.imread(host_image_path, cv2.IMREAD_GRAYSCALE)
        watermark_img = cv2.imread(watermark_image_path, cv2.IMREAD_GRAYSCALE)
        
        # Resize ảnh thủy vân để phù hợp với ảnh gốc
        watermark_img = cv2.resize(watermark_img, (host_img.shape[1], host_img.shape[0]))
        
        # Thực hiện DCT cho ảnh gốc
        dct_host = cv2.dct(np.float32(host_img))
        
        # Chuyển đổi ảnh thủy vân thành nhị phân
        watermark_binary = self.image_to_binary(watermark_img)
        
        # Nhúng ảnh thủy vân vào các hệ số DCT
        rows, cols = dct_host.shape
        alpha = 0.1  # Hệ số nhúng
        for i in range(len(watermark_binary)):
            row = (i // cols) % rows
            col = i % cols
            if watermark_binary[i] == '1':
                dct_host[row, col] = dct_host[row, col] * (1 + alpha)
            else:
                dct_host[row, col] = dct_host[row, col] * (1 - alpha)
        
        # Thực hiện IDCT
        watermarked_img = cv2.idct(dct_host)
        
        # Lưu ảnh đã nhúng thủy vân
        if not output_path.lower().endswith('.png'):
            output_path = output_path.rsplit('.', 1)[0] + '.png'
        cv2.imwrite(output_path, watermarked_img)
        return watermarked_img

    def dct_extract_image(self, watermarked_image, original_image_path, watermark_shape, output_path):
        """Trích xuất ảnh thủy vân từ ảnh đã nhúng sử dụng biến đổi DCT"""
        # Đọc ảnh gốc
        original = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
        
        # Kiểm tra nếu watermarked_image là đường dẫn file
        if isinstance(watermarked_image, str):
            watermarked = cv2.imread(watermarked_image, cv2.IMREAD_GRAYSCALE)
        else:
            watermarked = watermarked_image
        
        # Thực hiện DCT
        dct_original = cv2.dct(np.float32(original))
        dct_watermarked = cv2.dct(np.float32(watermarked))
        
        # Trích xuất ảnh thủy vân
        binary_message = ''
        rows, cols = dct_original.shape
        alpha = 0.1  # Hệ số nhúng
        
        # Số bit cần trích xuất = số pixel của ảnh thủy vân * 8
        num_bits = watermark_shape[0] * watermark_shape[1] * 8
        
        for i in range(num_bits):
            row = (i // cols) % rows
            col = i % cols
            ratio = dct_watermarked[row, col] / dct_original[row, col]
            if ratio > 1 + alpha/2:  # Ngưỡng so sánh
                binary_message += '1'
            else:
                binary_message += '0'
        
        # Chuyển đổi nhị phân thành ảnh
        extracted_img = self.binary_to_image(binary_message, watermark_shape)
        
        # Lưu ảnh đã trích xuất
        if not output_path.lower().endswith('.png'):
            output_path = output_path.rsplit('.', 1)[0] + '.png'
        cv2.imwrite(output_path, extracted_img)
        return extracted_img

    def apply_attacks(self, image, attack_type, params=None):
        """Áp dụng các loại tấn công khác nhau lên ảnh"""
        if params is None:
            params = {}
            
        if attack_type == 'noise':
            # Thêm nhiễu Gaussian
            mean = params.get('mean', 0)
            std = params.get('std', 10)
            noise = np.random.normal(mean, std, image.shape)
            return np.clip(image + noise, 0, 255).astype(np.uint8)
            
        elif attack_type == 'blur':
            # Làm mờ ảnh
            kernel_size = params.get('kernel_size', 5)
            return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
            
        elif attack_type == 'crop':
            # Cắt xén ảnh
            crop_percent = params.get('crop_percent', 0.1)
            h, w = image.shape
            crop_h = int(h * crop_percent)
            crop_w = int(w * crop_percent)
            cropped = image[crop_h:-crop_h, crop_w:-crop_w]
            return cv2.resize(cropped, (w, h))
            
        elif attack_type == 'rotate':
            # Xoay ảnh
            angle = params.get('angle', 5)
            h, w = image.shape
            center = (w//2, h//2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            return cv2.warpAffine(image, M, (w, h))
            
        elif attack_type == 'compress':
            # Nén ảnh
            quality = params.get('quality', 50)
            temp_path = 'temp_compressed.jpg'
            cv2.imwrite(temp_path, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            return cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
            
            else:
            raise ValueError(f"Loại tấn công không được hỗ trợ: {attack_type}")

    def evaluate_robustness(self, original_image, watermarked_image, watermark_image, attack_types):
        """Đánh giá độ bền vững của thủy vân với các loại tấn công khác nhau"""
        results = {}
        
        # Tính toán kích thước mới cho ảnh thủy vân (1/4 kích thước ảnh gốc)
        new_width = original_image.shape[1] // 4
        new_height = original_image.shape[0] // 4
        watermark_shape = (new_height, new_width)
        
        # Resize ảnh thủy vân
        watermark_img = cv2.resize(watermark_image, (new_width, new_height))
        
        for attack_type in attack_types:
            # Áp dụng tấn công
            if attack_type == 'noise':
                attacked_image = self.apply_attacks(watermarked_image, attack_type, {'std': 10})
            elif attack_type == 'blur':
                attacked_image = self.apply_attacks(watermarked_image, attack_type, {'kernel_size': 5})
            elif attack_type == 'crop':
                attacked_image = self.apply_attacks(watermarked_image, attack_type, {'crop_percent': 0.1})
            elif attack_type == 'rotate':
                attacked_image = self.apply_attacks(watermarked_image, attack_type, {'angle': 5})
            elif attack_type == 'compress':
                attacked_image = self.apply_attacks(watermarked_image, attack_type, {'quality': 50})
            
            # Trích xuất thủy vân từ ảnh bị tấn công
            output_path = f'output/{attack_type}_extracted.png'
            extracted_watermark = self.lsb_extract_image(attacked_image, watermark_shape, output_path)
            
            # Tính toán PSNR và SSIM giữa thủy vân gốc và thủy vân trích xuất
            psnr = self.calculate_psnr(watermark_img, extracted_watermark)
            ssim = self.calculate_ssim(watermark_img, extracted_watermark)
            
            results[attack_type] = {
                'psnr': psnr,
                'ssim': ssim
            }
        
        return results

    def calculate_psnr(self, img1, img2):
        """Tính PSNR giữa hai ảnh"""
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        max_pixel = 255.0
        psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
        return psnr

    def calculate_ssim(self, img1, img2):
        """Tính SSIM giữa hai ảnh"""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2
        
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)
        
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = cv2.GaussianBlur(img1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(img2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1_mu2
        
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        return np.mean(ssim_map) 