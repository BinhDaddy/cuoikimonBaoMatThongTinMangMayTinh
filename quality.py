import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image

class QualityAnalyzer:
    def __init__(self):
        pass

    def _ensure_numpy_array(self, img):
        """Chuyển đổi ảnh sang định dạng numpy array nếu cần"""
        if isinstance(img, Image.Image):
            return np.array(img)
        return img.astype(np.float32)

    def calculate_psnr(self, original_img, watermarked_img):
        """Tính PSNR (Peak Signal-to-Noise Ratio)"""
        original_img = self._ensure_numpy_array(original_img)
        watermarked_img = self._ensure_numpy_array(watermarked_img)
        return psnr(original_img, watermarked_img, data_range=255)

    def calculate_ssim(self, original_img, watermarked_img):
        """Tính SSIM (Structural Similarity Index)"""
        original_img = self._ensure_numpy_array(original_img)
        watermarked_img = self._ensure_numpy_array(watermarked_img)
        return ssim(original_img, watermarked_img, data_range=255)

    def calculate_mse(self, original_img, watermarked_img):
        """Tính MSE (Mean Squared Error)"""
        original_img = self._ensure_numpy_array(original_img)
        watermarked_img = self._ensure_numpy_array(watermarked_img)
        return np.mean((original_img - watermarked_img) ** 2)

    def analyze_image_quality(self, original_img, watermarked_img):
        """Phân tích chất lượng ảnh sau khi nhúng thủy vân"""
        results = {
            'psnr': self.calculate_psnr(original_img, watermarked_img),
            'ssim': self.calculate_ssim(original_img, watermarked_img),
            'mse': self.calculate_mse(original_img, watermarked_img)
        }
        return results

    def test_robustness(self, watermarked_img, original_img, watermark_text, extract_func):
        """Kiểm tra tính bền vững của thủy vân"""
        watermarked_img = self._ensure_numpy_array(watermarked_img)
        original_img = self._ensure_numpy_array(original_img)
        
        attacks = {
            'noise': self._add_noise,
            'compression': self._compress_image,
            'cropping': self._crop_image,
            'rotation': self._rotate_image
        }
        
        results = {}
        for attack_name, attack_func in attacks.items():
            attacked_img = attack_func(watermarked_img.copy())
            try:
                extracted_text = extract_func(attacked_img, original_img, len(watermark_text))
                results[attack_name] = {
                    'success': extracted_text == watermark_text,
                    'extracted_text': extracted_text
                }
            except Exception as e:
                results[attack_name] = {
                    'success': False,
                    'error': str(e)
                }
        return results

    def _add_noise(self, img, noise_level=0.1):
        """Thêm nhiễu Gaussian vào ảnh"""
        noise = np.random.normal(0, noise_level * 255, img.shape)
        noisy_img = img + noise
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

    def _compress_image(self, img, quality=50):
        """Nén ảnh JPEG"""
        _, compressed = cv2.imencode('.jpg', img.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, quality])
        return cv2.imdecode(compressed, cv2.IMREAD_GRAYSCALE)

    def _crop_image(self, img, crop_percent=0.1):
        """Cắt ảnh"""
        h, w = img.shape
        crop_h = int(h * crop_percent)
        crop_w = int(w * crop_percent)
        cropped = img[crop_h:-crop_h, crop_w:-crop_w]
        return cv2.resize(cropped, (w, h))

    def _rotate_image(self, img, angle=5):
        """Xoay ảnh"""
        h, w = img.shape
        center = (w // 2, h // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(img, rotation_matrix, (w, h)) 