# lise/encryption.py
import base64
import hashlib
from cryptography.fernet import Fernet
from lise.config import LISE_ENCRYPTION_KEY

def _get_fernet_key() -> bytes:
    """
    Deterministically generates a Fernet-compatible key from the
    master encryption key in the config.
    """
    # Use SHA256 to ensure the key is always 32 bytes long
    hashed_key = hashlib.sha256(LISE_ENCRYPTION_KEY.encode('utf-8')).digest()
    return base64.urlsafe_b64encode(hashed_key)

# Create a global Fernet instance to use for all operations
# This is safe and efficient.
_fernet = Fernet(_get_fernet_key())

def encrypt_key(plaintext_key: str) -> bytes:
    """
    Encrypts a plaintext API key.

    Args:
        plaintext_key: The third-party API key string to encrypt.

    Returns:
        The encrypted key as bytes, suitable for storing in a BLOB field.
    """
    if not isinstance(plaintext_key, str):
        raise TypeError("Key to be encrypted must be a string.")
        
    return _fernet.encrypt(plaintext_key.encode('utf-8'))


def decrypt_key(encrypted_key: bytes) -> str:
    """
    Decrypts an encrypted API key from the database.

    Args:
        encrypted_key: The encrypted key bytes from the database.

    Returns:
        The decrypted plaintext API key as a string.
    """
    if not isinstance(encrypted_key, bytes):
        raise TypeError("Encrypted key must be bytes.")
        
    decrypted_bytes = _fernet.decrypt(encrypted_key)
    return decrypted_bytes.decode('utf-8')