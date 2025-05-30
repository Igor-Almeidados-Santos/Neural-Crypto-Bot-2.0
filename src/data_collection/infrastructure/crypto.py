"""
Utilitários de criptografia para proteção de dados sensíveis.

Este módulo implementa funções para criptografia e descriptografia de dados,
como chaves de API e outros dados sensíveis armazenados pelo sistema.
"""
import base64
import hashlib
import os
from typing import Optional, Union, Dict, Any

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class CryptoService:
    """
    Serviço para criptografia e descriptografia de dados sensíveis.
    
    Implementa funções para proteger dados sensíveis como chaves de API,
    utilizando criptografia simétrica com Fernet.
    """
    
    def __init__(
        self, 
        master_key: Optional[str] = None,
        iterations: int = 100000,
        salt_size: int = 16
    ):
        """
        Inicializa o serviço de criptografia.
        
        Args:
            master_key: Chave mestra para derivar chaves de criptografia (opcional)
            iterations: Número de iterações para o PBKDF2
            salt_size: Tamanho do salt em bytes
        """
        # Se não for fornecida uma chave mestra, tenta obter da variável de ambiente
        self._master_key = master_key or os.environ.get('CRYPTO_MASTER_KEY')
        
        # Se ainda não houver chave, gera uma nova
        if not self._master_key:
            self._master_key = base64.b64encode(os.urandom(32)).decode('utf-8')
            print(f"⚠️ AVISO: Nenhuma chave mestra fornecida. Gerada nova chave: {self._master_key}")
            print("Por favor, armazene esta chave em um local seguro e defina como variável de ambiente CRYPTO_MASTER_KEY.")
        
        self._iterations = iterations
        self._salt_size = salt_size
        self._fernet_instances: Dict[str, Fernet] = {}
    
    def _get_fernet(self, salt: bytes) -> Fernet:
        """
        Obtém ou cria uma instância Fernet para um determinado salt.
        
        Args:
            salt: Salt para derivação da chave
            
        Returns:
            Fernet: Instância Fernet para criptografia/descriptografia
        """
        # Usa o salt como chave de cache
        salt_hex = salt.hex()
        
        if salt_hex not in self._fernet_instances:
            # Deriva uma chave a partir da chave mestra e do salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=self._iterations,
            )
            key = base64.urlsafe_b64encode(kdf.derive(self._master_key.encode()))
            self._fernet_instances[salt_hex] = Fernet(key)
            
        return self._fernet_instances[salt_hex]
    
    def encrypt(self, data: Union[str, bytes]) -> str:
        """
        Criptografa dados usando a chave mestra.
        
        Args:
            data: Dados a serem criptografados (string ou bytes)
            
        Returns:
            str: Dados criptografados em formato base64
        """
        # Converte dados para bytes se for string
        if isinstance(data, str):
            data = data.encode('utf-8')
        
        # Gera um salt aleatório
        salt = os.urandom(self._salt_size)
        
        # Obtém a instância Fernet para este salt
        f = self._get_fernet(salt)
        
        # Criptografa os dados
        encrypted_data = f.encrypt(data)
        
        # Combina salt + dados criptografados e codifica em base64
        result = base64.b64encode(salt + encrypted_data).decode('utf-8')
        
        return result
    
    def decrypt(self, encrypted_data: str) -> bytes:
        """
        Descriptografa dados usando a chave mestra.
        
        Args:
            encrypted_data: Dados criptografados em formato base64
            
        Returns:
            bytes: Dados descriptografados
            
        Raises:
            ValueError: Se os dados estiverem corrompidos ou a chave for inválida
        """
        # Decodifica o base64
        raw_data = base64.b64decode(encrypted_data)
        
        # Extrai o salt e os dados criptografados
        salt = raw_data[:self._salt_size]
        encrypted_payload = raw_data[self._salt_size:]
        
        # Obtém a instância Fernet para este salt
        f = self._get_fernet(salt)
        
        # Descriptografa os dados
        decrypted_data = f.decrypt(encrypted_payload)
        
        return decrypted_data
    
    def decrypt_to_string(self, encrypted_data: str) -> str:
        """
        Descriptografa dados e retorna como string.
        
        Args:
            encrypted_data: Dados criptografados em formato base64
            
        Returns:
            str: Dados descriptografados como string
            
        Raises:
            ValueError: Se os dados estiverem corrompidos ou a chave for inválida
        """
        return self.decrypt(encrypted_data).decode('utf-8')
    
    def encrypt_dict(self, data: Dict[str, Any], sensitive_keys: Optional[list] = None) -> Dict[str, Any]:
        """
        Criptografa valores sensíveis em um dicionário.
        
        Args:
            data: Dicionário a ser processado
            sensitive_keys: Lista de chaves a serem criptografadas (opcional)
            
        Returns:
            Dict[str, Any]: Dicionário com valores sensíveis criptografados
        """
        # Lista padrão de chaves sensíveis
        default_sensitive_keys = [
            'api_key', 'apiKey', 'api_secret', 'apiSecret', 'secret', 
            'password', 'passphrase', 'private_key', 'privateKey',
            'token', 'access_token', 'refresh_token'
        ]
        
        sensitive_keys = sensitive_keys or default_sensitive_keys
        result = {}
        
        for key, value in data.items():
            if key in sensitive_keys and value:
                # Criptografa o valor se for uma chave sensível
                if isinstance(value, str):
                    result[key] = self.encrypt(value)
                else:
                    result[key] = value
            elif isinstance(value, dict):
                # Recursivamente processa subdicionários
                result[key] = self.encrypt_dict(value, sensitive_keys)
            else:
                # Mantém valores não sensíveis inalterados
                result[key] = value
                
        return result
    
    def decrypt_dict(self, data: Dict[str, Any], encrypted_keys: Optional[list] = None) -> Dict[str, Any]:
        """
        Descriptografa valores sensíveis em um dicionário.
        
        Args:
            data: Dicionário a ser processado
            encrypted_keys: Lista de chaves a serem descriptografadas (opcional)
            
        Returns:
            Dict[str, Any]: Dicionário com valores sensíveis descriptografados
        """
        # Lista padrão de chaves sensíveis
        default_encrypted_keys = [
            'api_key', 'apiKey', 'api_secret', 'apiSecret', 'secret', 
            'password', 'passphrase', 'private_key', 'privateKey',
            'token', 'access_token', 'refresh_token'
        ]
        
        encrypted_keys = encrypted_keys or default_encrypted_keys
        result = {}
        
        for key, value in data.items():
            if key in encrypted_keys and value and isinstance(value, str):
                try:
                    # Tenta descriptografar o valor
                    result[key] = self.decrypt_to_string(value)
                except Exception:
                    # Se falhar, assume que o valor não está criptografado
                    result[key] = value
            elif isinstance(value, dict):
                # Recursivamente processa subdicionários
                result[key] = self.decrypt_dict(value, encrypted_keys)
            else:
                # Mantém outros valores inalterados
                result[key] = value
                
        return result
    
    def hash_password(self, password: str) -> str:
        """
        Gera um hash seguro para uma senha.
        
        Args:
            password: Senha a ser hasheada
            
        Returns:
            str: Hash da senha em formato base64
        """
        # Gera um salt aleatório
        salt = os.urandom(self._salt_size)
        
        # Gera o hash usando PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self._iterations,
        )
        password_hash = kdf.derive(password.encode())
        
        # Combina salt + hash e codifica em base64
        result = base64.b64encode(salt + password_hash).decode('utf-8')
        
        return result
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verifica se uma senha corresponde a um hash.
        
        Args:
            password: Senha a ser verificada
            password_hash: Hash da senha em formato base64
            
        Returns:
            bool: True se a senha corresponder ao hash, False caso contrário
        """
        try:
            # Decodifica o base64
            raw_data = base64.b64decode(password_hash)
            
            # Extrai o salt e o hash
            salt = raw_data[:self._salt_size]
            stored_hash = raw_data[self._salt_size:]
            
            # Gera o hash da senha fornecida usando o mesmo salt
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=self._iterations,
            )
            password_hash = kdf.derive(password.encode())
            
            # Compara os hashes
            return password_hash == stored_hash
            
        except Exception:
            return False