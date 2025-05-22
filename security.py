import hashlib
from cryptography.fernet import Fernet
import base64
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

class SecurityManager:
    def __init__(self, password):
        """Khởi tạo SecurityManager với password"""
        self.password = password
        self.key = self._generate_key(password)
        self.cipher_suite = Fernet(self.key)

    def _generate_key(self, password):
        """Tạo key từ password sử dụng PBKDF2"""
        salt = b'fixed_salt'  # Trong thực tế nên sử dụng salt ngẫu nhiên
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
        return key

    def encrypt_message(self, message):
        """Mã hóa thông điệp trước khi nhúng"""
        encrypted_message = self.cipher_suite.encrypt(message.encode())
        return encrypted_message

    def decrypt_message(self, encrypted_message):
        """Giải mã thông điệp sau khi trích xuất"""
        decrypted_message = self.cipher_suite.decrypt(encrypted_message)
        return decrypted_message.decode()

    def generate_hash(self, data):
        """Tạo hash để xác thực tính toàn vẹn"""
        return hashlib.sha256(data).hexdigest()

    def verify_hash(self, data, hash_value):
        """Xác thực tính toàn vẹn của dữ liệu"""
        return self.generate_hash(data) == hash_value

    def create_watermark_package(self, message):
        """Tạo gói thủy vân bao gồm thông điệp đã mã hóa và hash"""
        encrypted_message = self.encrypt_message(message)
        message_hash = self.generate_hash(encrypted_message)
        return {
            'encrypted_message': encrypted_message,
            'hash': message_hash
        }

    def verify_watermark_package(self, package):
        """Xác thực gói thủy vân"""
        if not self.verify_hash(package['encrypted_message'], package['hash']):
            raise ValueError("Thủy vân đã bị thay đổi!")
        return self.decrypt_message(package['encrypted_message']) 