"""
Serviço de criptografia para proteção de dados sensíveis.

Este módulo implementa criptografia robusta para proteger credenciais de API,
chaves privadas e outros dados sensíveis utilizados pelo sistema de trading.
"""
import os
import base64
import hashlib
import hmac
import secrets
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import json

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import bcrypt
import jwt

logger = logging.getLogger(__name__)


class EncryptionAlgorithm(Enum):
    """Algoritmos de criptografia disponíveis."""
    FERNET = "fernet"  # Criptografia simétrica com Fernet
    AES_256_GCM = "aes_256_gcm"  # AES-256 com GCM
    RSA_2048 = "rsa_2048"  # RSA 2048 bits
    RSA_4096 = "rsa_4096"  # RSA 4096 bits


class HashAlgorithm(Enum):
    """Algoritmos de hash disponíveis."""
    SHA256 = "sha256"
    SHA512 = "sha512"
    BCRYPT = "bcrypt"
    SCRYPT = "scrypt"


@dataclass
class CryptoConfig:
    """Configuração para operações criptográficas."""
    default_algorithm: EncryptionAlgorithm = EncryptionAlgorithm.FERNET
    key_derivation_iterations: int = 100000
    salt_length: int = 32
    enable_key_rotation: bool = True
    key_rotation_days: int = 30
    max_key_age_days: int = 90


@dataclass
class EncryptedData:
    """Dados criptografados com metadados."""
    ciphertext: str  # Base64 encoded
    algorithm: str
    key_id: Optional[str] = None
    salt: Optional[str] = None  # Base64 encoded
    iv: Optional[str] = None  # Base64 encoded
    created_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return {
            'ciphertext': self.ciphertext,
            'algorithm': self.algorithm,
            'key_id': self.key_id,
            'salt': self.salt,
            'iv': self.iv,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EncryptedData':
        """Cria instância a partir de dicionário."""
        created_at = None
        if data.get('created_at'):
            created_at = datetime.fromisoformat(data['created_at'])
        
        return cls(
            ciphertext=data['ciphertext'],
            algorithm=data['algorithm'],
            key_id=data.get('key_id'),
            salt=data.get('salt'),
            iv=data.get('iv'),
            created_at=created_at,
            metadata=data.get('metadata')
        )


class KeyManager:
    """
    Gerenciador de chaves criptográficas.
    
    Responsável por gerar, armazenar e rotacionar chaves de criptografia
    de forma segura.
    """
    
    def __init__(self, config: CryptoConfig):
        self.config = config
        self._keys: Dict[str, bytes] = {}
        self._key_metadata: Dict[str, Dict[str, Any]] = {}
        self._master_key: Optional[bytes] = None
        self._current_key_id: Optional[str] = None
    
    def set_master_key(self, master_key: Union[str, bytes]) -> None:
        """
        Define a chave mestra para derivação de outras chaves.
        
        Args:
            master_key: Chave mestra como string ou bytes
        """
        if isinstance(master_key, str):
            master_key = master_key.encode('utf-8')
        
        self._master_key = master_key
        logger.info("Chave mestra configurada")
    
    def generate_key(
        self,
        algorithm: EncryptionAlgorithm = None,
        key_id: Optional[str] = None
    ) -> str:
        """
        Gera uma nova chave criptográfica.
        
        Args:
            algorithm: Algoritmo para qual gerar a chave
            key_id: ID personalizado para a chave
            
        Returns:
            ID da chave gerada
        """
        algorithm = algorithm or self.config.default_algorithm
        key_id = key_id or self._generate_key_id()
        
        if algorithm == EncryptionAlgorithm.FERNET:
            key = Fernet.generate_key()
        elif algorithm in [EncryptionAlgorithm.AES_256_GCM]:
            key = secrets.token_bytes(32)  # 256 bits
        else:
            raise ValueError(f"Algoritmo não suportado para geração de chave: {algorithm}")
        
        self._keys[key_id] = key
        self._key_metadata[key_id] = {
            'algorithm': algorithm.value,
            'created_at': datetime.utcnow(),
            'last_used': datetime.utcnow(),
            'usage_count': 0
        }
        
        if self._current_key_id is None:
            self._current_key_id = key_id
        
        logger.info(f"Chave gerada: {key_id} ({algorithm.value})")
        return key_id
    
    def derive_key_from_password(
        self,
        password: str,
        salt: Optional[bytes] = None,
        algorithm: HashAlgorithm = HashAlgorithm.SCRYPT
    ) -> Tuple[bytes, bytes]:
        """
        Deriva uma chave a partir de uma senha.
        
        Args:
            password: Senha para derivação
            salt: Salt opcional (será gerado se não fornecido)
            algorithm: Algoritmo de derivação
            
        Returns:
            Tupla (chave_derivada, salt_usado)
        """
        if salt is None:
            salt = secrets.token_bytes(self.config.salt_length)
        
        password_bytes = password.encode('utf-8')
        
        if algorithm == HashAlgorithm.SCRYPT:
            kdf = Scrypt(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                n=2**14,
                r=8,
                p=1,
                backend=default_backend()
            )
        else:  # PBKDF2
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=self.config.key_derivation_iterations,
                backend=default_backend()
            )
        
        key = kdf.derive(password_bytes)
        return key, salt
    
    def get_key(self, key_id: str) -> Optional[bytes]:
        """
        Obtém uma chave pelo ID.
        
        Args:
            key_id: ID da chave
            
        Returns:
            Chave em bytes ou None se não encontrada
        """
        key = self._keys.get(key_id)
        if key and key_id in self._key_metadata:
            self._key_metadata[key_id]['last_used'] = datetime.utcnow()
            self._key_metadata[key_id]['usage_count'] += 1
        
        return key
    
    def get_current_key(self) -> Optional[Tuple[str, bytes]]:
        """
        Obtém a chave atual para criptografia.
        
        Returns:
            Tupla (key_id, key) ou None se não houver chave
        """
        if self._current_key_id:
            key = self.get_key(self._current_key_id)
            if key:
                return self._current_key_id, key
        
        return None
    
    def rotate_keys(self) -> List[str]:
        """
        Rotaciona chaves antigas baseado na configuração.
        
        Returns:
            Lista de IDs das chaves rotacionadas
        """
        if not self.config.enable_key_rotation:
            return []
        
        now = datetime.utcnow()
        rotated_keys = []
        
        for key_id, metadata in self._key_metadata.items():
            age = now - metadata['created_at']
            
            if age.days >= self.config.key_rotation_days:
                # Gera nova chave para substituir
                algorithm = EncryptionAlgorithm(metadata['algorithm'])
                new_key_id = self.generate_key(algorithm)
                
                # Atualiza chave atual se necessário
                if self._current_key_id == key_id:
                    self._current_key_id = new_key_id
                
                rotated_keys.append(key_id)
        
        return rotated_keys
    
    def cleanup_old_keys(self) -> List[str]:
        """
        Remove chaves muito antigas baseado na configuração.
        
        Returns:
            Lista de IDs das chaves removidas
        """
        now = datetime.utcnow()
        removed_keys = []
        
        for key_id, metadata in list(self._key_metadata.items()):
            age = now - metadata['created_at']
            
            if age.days >= self.config.max_key_age_days:
                del self._keys[key_id]
                del self._key_metadata[key_id]
                removed_keys.append(key_id)
        
        return removed_keys
    
    def _generate_key_id(self) -> str:
        """Gera um ID único para chave."""
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        random_suffix = secrets.token_hex(4)
        return f"key_{timestamp}_{random_suffix}"
    
    def get_key_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas das chaves."""
        return {
            'total_keys': len(self._keys),
            'current_key_id': self._current_key_id,
            'keys_metadata': {
                key_id: {
                    **metadata,
                    'created_at': metadata['created_at'].isoformat(),
                    'last_used': metadata['last_used'].isoformat()
                }
                for key_id, metadata in self._key_metadata.items()
            }
        }


class CryptoService:
    """
    Serviço principal de criptografia para o sistema de trading.
    
    Implementa criptografia robusta para proteger dados sensíveis,
    incluindo credenciais de API, chaves privadas e configurações.
    """
    
    def __init__(self, config: CryptoConfig = None):
        """
        Inicializa o serviço de criptografia.
        
        Args:
            config: Configuração criptográfica
        """
        self.config = config or CryptoConfig()
        self.key_manager = KeyManager(self.config)
        
        # Inicializa com chave padrão se disponível no ambiente
        master_key = os.getenv('CRYPTO_MASTER_KEY')
        if master_key:
            self.key_manager.set_master_key(master_key)
            self.key_manager.generate_key()
        
        logger.info("Serviço de criptografia inicializado")
    
    def encrypt(
        self,
        plaintext: Union[str, bytes, Dict[str, Any]],
        algorithm: EncryptionAlgorithm = None,
        key_id: Optional[str] = None
    ) -> EncryptedData:
        """
        Criptografa dados.
        
        Args:
            plaintext: Dados a serem criptografados
            algorithm: Algoritmo de criptografia
            key_id: ID da chave específica (opcional)
            
        Returns:
            Dados criptografados com metadados
        """
        algorithm = algorithm or self.config.default_algorithm
        
        # Converte dados para bytes se necessário
        if isinstance(plaintext, dict):
            plaintext = json.dumps(plaintext)
        if isinstance(plaintext, str):
            plaintext = plaintext.encode('utf-8')
        
        if algorithm == EncryptionAlgorithm.FERNET:
            return self._encrypt_fernet(plaintext, key_id)
        elif algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._encrypt_aes_gcm(plaintext, key_id)
        else:
            raise ValueError(f"Algoritmo de criptografia não implementado: {algorithm}")
    
    def decrypt(self, encrypted_data: EncryptedData) -> bytes:
        """
        Descriptografa dados.
        
        Args:
            encrypted_data: Dados criptografados
            
        Returns:
            Dados descriptografados em bytes
        """
        algorithm = EncryptionAlgorithm(encrypted_data.algorithm)
        
        if algorithm == EncryptionAlgorithm.FERNET:
            return self._decrypt_fernet(encrypted_data)
        elif algorithm == EncryptionAlgorithm.AES_256_GCM:
            return self._decrypt_aes_gcm(encrypted_data)
        else:
            raise ValueError(f"Algoritmo de descriptografia não implementado: {algorithm}")
    
    def encrypt_string(self, plaintext: str, algorithm: EncryptionAlgorithm = None) -> str:
        """
        Criptografa uma string e retorna como JSON.
        
        Args:
            plaintext: String a ser criptografada
            algorithm: Algoritmo de criptografia
            
        Returns:
            String JSON com dados criptografados
        """
        encrypted = self.encrypt(plaintext, algorithm)
        return json.dumps(encrypted.to_dict())
    
    def decrypt_string(self, encrypted_json: str) -> str:
        """
        Descriptografa dados de uma string JSON.
        
        Args:
            encrypted_json: String JSON com dados criptografados
            
        Returns:
            String descriptografada
        """
        encrypted_dict = json.loads(encrypted_json)
        encrypted_data = EncryptedData.from_dict(encrypted_dict)
        decrypted_bytes = self.decrypt(encrypted_data)
        return decrypted_bytes.decode('utf-8')
    
    def encrypt_dict(self, data: Dict[str, Any]) -> Dict[str, str]:
        """
        Criptografa valores sensíveis em um dicionário.
        
        Args:
            data: Dicionário com dados a serem criptografados
            
        Returns:
            Dicionário com valores criptografados
        """
        encrypted_dict = {}
        
        for key, value in data.items():
            if self._is_sensitive_key(key):
                encrypted_value = self.encrypt_string(str(value))
                encrypted_dict[key] = encrypted_value
            else:
                encrypted_dict[key] = value
        
        return encrypted_dict
    
    def decrypt_dict(self, encrypted_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Descriptografa valores em um dicionário.
        
        Args:
            encrypted_data: Dicionário com valores criptografados
            
        Returns:
            Dicionário com valores descriptografados
        """
        decrypted_dict = {}
        
        for key, value in encrypted_data.items():
            if self._is_sensitive_key(key) and isinstance(value, str):
                try:
                    # Tenta descriptografar se parecer ser dados criptografados
                    if value.startswith('{') and 'ciphertext' in value:
                        decrypted_value = self.decrypt_string(value)
                        decrypted_dict[key] = decrypted_value
                    else:
                        decrypted_dict[key] = value
                except Exception:
                    # Se falhar, mantém valor original
                    decrypted_dict[key] = value
            else:
                decrypted_dict[key] = value
        
        return decrypted_dict
    
    def hash_password(self, password: str, algorithm: HashAlgorithm = HashAlgorithm.BCRYPT) -> str:
        """
        Gera hash seguro de senha.
        
        Args:
            password: Senha a ser hasheada
            algorithm: Algoritmo de hash
            
        Returns:
            Hash da senha
        """
        if algorithm == HashAlgorithm.BCRYPT:
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            return hashed.decode('utf-8')
        
        elif algorithm == HashAlgorithm.SHA256:
            salt = secrets.token_hex(16)
            hashed = hashlib.sha256((password + salt).encode('utf-8')).hexdigest()
            return f"sha256${salt}${hashed}"
        
        elif algorithm == HashAlgorithm.SHA512:
            salt = secrets.token_hex(16)
            hashed = hashlib.sha512((password + salt).encode('utf-8')).hexdigest()
            return f"sha512${salt}${hashed}"
        
        else:
            raise ValueError(f"Algoritmo de hash não implementado: {algorithm}")
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """
        Verifica se a senha corresponde ao hash.
        
        Args:
            password: Senha em texto claro
            hashed: Hash da senha
            
        Returns:
            True se a senha corresponde
        """
        try:
            if hashed.startswith('$2b$'):  # bcrypt
                return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
            
            elif hashed.startswith('sha256$'):
                _, salt, stored_hash = hashed.split('$')
                computed_hash = hashlib.sha256((password + salt).encode('utf-8')).hexdigest()
                return hmac.compare_digest(stored_hash, computed_hash)
            
            elif hashed.startswith('sha512$'):
                _, salt, stored_hash = hashed.split('$')
                computed_hash = hashlib.sha512((password + salt).encode('utf-8')).hexdigest()
                return hmac.compare_digest(stored_hash, computed_hash)
            
            return False
            
        except Exception as e:
            logger.error(f"Erro na verificação de senha: {e}")
            return False
    
    def generate_api_signature(
        self,
        secret_key: str,
        message: str,
        algorithm: str = 'sha256'
    ) -> str:
        """
        Gera assinatura HMAC para APIs de exchanges.
        
        Args:
            secret_key: Chave secreta da API
            message: Mensagem a ser assinada
            algorithm: Algoritmo de hash
            
        Returns:
            Assinatura em hexadecimal
        """
        if algorithm == 'sha256':
            signature = hmac.new(
                secret_key.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
        elif algorithm == 'sha512':
            signature = hmac.new(
                secret_key.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha512
            ).hexdigest()
        else:
            raise ValueError(f"Algoritmo não suportado: {algorithm}")
        
        return signature
    
    def generate_jwt_token(
        self,
        payload: Dict[str, Any],
        secret_key: Optional[str] = None,
        algorithm: str = 'HS256',
        expires_in: timedelta = timedelta(hours=24)
    ) -> str:
        """
        Gera token JWT.
        
        Args:
            payload: Dados do payload
            secret_key: Chave secreta (opcional)
            algorithm: Algoritmo de assinatura
            expires_in: Tempo de expiração
            
        Returns:
            Token JWT
        """
        if secret_key is None:
            current_key = self.key_manager.get_current_key()
            if not current_key:
                raise ValueError("Nenhuma chave disponível para JWT")
            secret_key = base64.b64encode(current_key[1]).decode('utf-8')
        
        # Adiciona tempo de expiração
        payload = payload.copy()
        payload['exp'] = datetime.utcnow() + expires_in
        payload['iat'] = datetime.utcnow()
        
        return jwt.encode(payload, secret_key, algorithm=algorithm)
    
    def verify_jwt_token(
        self,
        token: str,
        secret_key: Optional[str] = None,
        algorithm: str = 'HS256'
    ) -> Dict[str, Any]:
        """
        Verifica e decodifica token JWT.
        
        Args:
            token: Token JWT
            secret_key: Chave secreta (opcional)
            algorithm: Algoritmo de verificação
            
        Returns:
            Payload decodificado
        """
        if secret_key is None:
            current_key = self.key_manager.get_current_key()
            if not current_key:
                raise ValueError("Nenhuma chave disponível para JWT")
            secret_key = base64.b64encode(current_key[1]).decode('utf-8')
        
        return jwt.decode(token, secret_key, algorithms=[algorithm])
    
    def _encrypt_fernet(self, plaintext: bytes, key_id: Optional[str] = None) -> EncryptedData:
        """Criptografa usando Fernet."""
        if key_id:
            key = self.key_manager.get_key(key_id)
            if not key:
                raise ValueError(f"Chave não encontrada: {key_id}")
        else:
            current_key = self.key_manager.get_current_key()
            if not current_key:
                raise ValueError("Nenhuma chave disponível")
            key_id, key = current_key
        
        fernet = Fernet(key)
        ciphertext = fernet.encrypt(plaintext)
        
        return EncryptedData(
            ciphertext=base64.b64encode(ciphertext).decode('utf-8'),
            algorithm=EncryptionAlgorithm.FERNET.value,
            key_id=key_id,
            created_at=datetime.utcnow()
        )
    
    def _decrypt_fernet(self, encrypted_data: EncryptedData) -> bytes:
        """Descriptografa usando Fernet."""
        key = self.key_manager.get_key(encrypted_data.key_id)
        if not key:
            raise ValueError(f"Chave não encontrada: {encrypted_data.key_id}")
        
        fernet = Fernet(key)
        ciphertext = base64.b64decode(encrypted_data.ciphertext.encode('utf-8'))
        
        return fernet.decrypt(ciphertext)
    
    def _encrypt_aes_gcm(self, plaintext: bytes, key_id: Optional[str] = None) -> EncryptedData:
        """Criptografa usando AES-256-GCM."""
        if key_id:
            key = self.key_manager.get_key(key_id)
            if not key:
                raise ValueError(f"Chave não encontrada: {key_id}")
        else:
            current_key = self.key_manager.get_current_key()
            if not current_key:
                raise ValueError("Nenhuma chave disponível")
            key_id, key = current_key
        
        # Gera IV aleatório
        iv = secrets.token_bytes(12)  # 96 bits para GCM
        
        # Criptografa
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv), backend=default_backend())
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(plaintext) + encryptor.finalize()
        
        # Adiciona tag de autenticação
        final_ciphertext = ciphertext + encryptor.tag
        
        return EncryptedData(
            ciphertext=base64.b64encode(final_ciphertext).decode('utf-8'),
            algorithm=EncryptionAlgorithm.AES_256_GCM.value,
            key_id=key_id,
            iv=base64.b64encode(iv).decode('utf-8'),
            created_at=datetime.utcnow()
        )
    
    def _decrypt_aes_gcm(self, encrypted_data: EncryptedData) -> bytes:
        """Descriptografa usando AES-256-GCM."""
        key = self.key_manager.get_key(encrypted_data.key_id)
        if not key:
            raise ValueError(f"Chave não encontrada: {encrypted_data.key_id}")
        
        iv = base64.b64decode(encrypted_data.iv.encode('utf-8'))
        ciphertext_with_tag = base64.b64decode(encrypted_data.ciphertext.encode('utf-8'))
        
        # Separa ciphertext da tag
        ciphertext = ciphertext_with_tag[:-16]
        tag = ciphertext_with_tag[-16:]
        
        # Descriptografa
        cipher = Cipher(algorithms.AES(key), modes.GCM(iv, tag), backend=default_backend())
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext
    
    def _is_sensitive_key(self, key: str) -> bool:
        """Identifica se uma chave contém dados sensíveis."""
        sensitive_keys = {
            'api_key', 'api_secret', 'api_passphrase', 'private_key',
            'password', 'secret', 'token', 'credential', 'auth'
        }
        
        key_lower = key.lower()
        return any(sensitive in key_lower for sensitive in sensitive_keys)
    
    def rotate_keys(self) -> Dict[str, List[str]]:
        """
        Executa rotação de chaves.
        
        Returns:
            Dicionário com chaves rotacionadas e removidas
        """
        rotated = self.key_manager.rotate_keys()
        removed = self.key_manager.cleanup_old_keys()
        
        logger.info(f"Rotação de chaves: {len(rotated)} rotacionadas, {len(removed)} removidas")
        
        return {
            'rotated': rotated,
            'removed': removed
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do serviço de criptografia."""
        return {
            'config': {
                'default_algorithm': self.config.default_algorithm.value,
                'key_rotation_enabled': self.config.enable_key_rotation,
                'key_rotation_days': self.config.key_rotation_days,
                'max_key_age_days': self.config.max_key_age_days
            },
            'keys': self.key_manager.get_key_stats()
        }


# Instância global padrão
_default_crypto_service: Optional[CryptoService] = None


def get_crypto_service() -> CryptoService:
    """Obtém instância global do serviço de criptografia."""
    global _default_crypto_service
    if _default_crypto_service is None:
        _default_crypto_service = CryptoService()
    return _default_crypto_service


def encrypt_api_credentials(credentials: Dict[str, str]) -> Dict[str, str]:
    """Função de conveniência para criptografar credenciais de API."""
    crypto_service = get_crypto_service()
    return crypto_service.encrypt_dict(credentials)


def decrypt_api_credentials(encrypted_credentials: Dict[str, str]) -> Dict[str, str]:
    """Função de conveniência para descriptografar credenciais de API."""
    crypto_service = get_crypto_service()
    return crypto_service.decrypt_dict(encrypted_credentials)