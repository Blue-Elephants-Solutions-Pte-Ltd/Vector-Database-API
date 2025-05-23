from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import padding
import os
from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env.

def decrypt_api_key(api_key):
    
    # Get encryption key and IV from environment variables or use defaults
    # Get encryption key and IV from environment variables or use defaults
    key = os.getenv('AES_ENC_KEY').encode('utf-8')
    iv = os.getenv('AES_IV').encode('utf-8')
    
    
    # Create cipher
    cipher = Cipher(
        algorithms.AES(key),
        modes.CBC(iv),
        backend=default_backend()
    )
    
    # Create decryptor
    decryptor = cipher.decryptor()
    
    # Decrypt the API key
    encrypted_data = bytes.fromhex(api_key)
    decrypted_data = decryptor.update(encrypted_data) + decryptor.finalize()
    
    # Create unpadder
    unpadder = padding.PKCS7(algorithms.AES.block_size).unpadder()
    
    # Remove padding
    unpadded_data = unpadder.update(decrypted_data) + unpadder.finalize()
    
    # Decode to UTF-8
    api_key = unpadded_data.decode('utf-8')
    print(f"Decrypted API Key: {api_key}")
    return api_key
