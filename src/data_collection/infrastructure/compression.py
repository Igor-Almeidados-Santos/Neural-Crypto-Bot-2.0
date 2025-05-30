"""
Utilitários de compressão para armazenamento eficiente de dados.

Este módulo implementa funções para compressão e descompressão de dados,
permitindo armazenamento mais eficiente de grandes volumes de dados de mercado.
"""
import json
import zlib
import lzma
import bz2
from typing import Any, Dict, List, Union, Optional


class CompressionService:
    """
    Serviço para compressão e descompressão de dados.
    
    Implementa funções para compactar e descompactar dados em vários formatos,
    otimizando o armazenamento e transferência de grandes volumes de dados.
    """
    
    # Constantes para tipos de compressão
    ZLIB = 'zlib'
    LZMA = 'lzma'
    BZ2 = 'bz2'
    
    def __init__(self, default_method: str = ZLIB, compression_level: int = 6):
        """
        Inicializa o serviço de compressão.
        
        Args:
            default_method: Método de compressão padrão (zlib, lzma ou bz2)
            compression_level: Nível de compressão (1-9, onde 9 é máximo)
        """
        if default_method not in [self.ZLIB, self.LZMA, self.BZ2]:
            raise ValueError(f"Método de compressão inválido: {default_method}")
            
        if compression_level < 1 or compression_level > 9:
            raise ValueError(f"Nível de compressão inválido: {compression_level}. Deve estar entre 1 e 9.")
            
        self.default_method = default_method
        self.compression_level = compression_level
    
    def compress(self, data: bytes, method: Optional[str] = None) -> bytes:
        """
        Comprime dados binários.
        
        Args:
            data: Dados a serem comprimidos
            method: Método de compressão (zlib, lzma ou bz2)
            
        Returns:
            bytes: Dados comprimidos
        """
        method = method or self.default_method
        
        if method == self.ZLIB:
            return zlib.compress(data, level=self.compression_level)
        elif method == self.LZMA:
            return lzma.compress(data, preset=self.compression_level)
        elif method == self.BZ2:
            return bz2.compress(data, compresslevel=self.compression_level)
        else:
            raise ValueError(f"Método de compressão inválido: {method}")
    
    def decompress(self, compressed_data: bytes, method: Optional[str] = None) -> bytes:
        """
        Descomprime dados binários.
        
        Args:
            compressed_data: Dados comprimidos
            method: Método de compressão usado (zlib, lzma ou bz2)
            
        Returns:
            bytes: Dados descomprimidos
        """
        method = method or self.default_method
        
        if method == self.ZLIB:
            return zlib.decompress(compressed_data)
        elif method == self.LZMA:
            return lzma.decompress(compressed_data)
        elif method == self.BZ2:
            return bz2.decompress(compressed_data)
        else:
            raise ValueError(f"Método de compressão inválido: {method}")
    
    def compress_string(self, data: str, method: Optional[str] = None) -> bytes:
        """
        Comprime uma string.
        
        Args:
            data: String a ser comprimida
            method: Método de compressão (zlib, lzma ou bz2)
            
        Returns:
            bytes: Dados comprimidos
        """
        return self.compress(data.encode('utf-8'), method)
    
    def decompress_to_string(self, compressed_data: bytes, method: Optional[str] = None) -> str:
        """
        Descomprime dados para uma string.
        
        Args:
            compressed_data: Dados comprimidos
            method: Método de compressão usado (zlib, lzma ou bz2)
            
        Returns:
            str: String descomprimida
        """
        return self.decompress(compressed_data, method).decode('utf-8')
    
    def compress_json(self, data: Union[Dict, List, Any], method: Optional[str] = None) -> bytes:
        """
        Comprime dados JSON.
        
        Args:
            data: Dados a serem convertidos para JSON e comprimidos
            method: Método de compressão (zlib, lzma ou bz2)
            
        Returns:
            bytes: Dados JSON comprimidos
        """
        json_str = json.dumps(data)
        return self.compress_string(json_str, method)
    
    def decompress_json(self, compressed_data: bytes, method: Optional[str] = None) -> Any:
        """
        Descomprime dados JSON.
        
        Args:
            compressed_data: Dados JSON comprimidos
            method: Método de compressão usado (zlib, lzma ou bz2)
            
        Returns:
            Any: Dados JSON descomprimidos e parseados
        """
        json_str = self.decompress_to_string(compressed_data, method)
        return json.loads(json_str)
    
    def should_compress(self, data: bytes, threshold_bytes: int = 1024) -> bool:
        """
        Determina se vale a pena comprimir os dados com base no tamanho.
        
        Args:
            data: Dados a serem avaliados
            threshold_bytes: Tamanho mínimo em bytes para comprimir
            
        Returns:
            bool: True se os dados devem ser comprimidos, False caso contrário
        """
        return len(data) >= threshold_bytes
    
    def compress_efficient(self, data: bytes, threshold_bytes: int = 1024) -> Dict[str, Any]:
        """
        Comprime dados apenas se for eficiente fazê-lo.
        
        Args:
            data: Dados a serem comprimidos
            threshold_bytes: Tamanho mínimo em bytes para comprimir
            
        Returns:
            Dict[str, Any]: Dicionário contendo:
                - compressed: True se os dados foram comprimidos, False caso contrário
                - method: Método de compressão usado
                - data: Dados comprimidos ou originais
        """
        if not self.should_compress(data, threshold_bytes):
            return {
                'compressed': False,
                'method': None,
                'data': data
            }
        
        # Tenta comprimir com o método padrão
        compressed = self.compress(data)
        
        # Verifica se a compressão valeu a pena (pelo menos 10% menor)
        if len(compressed) < len(data) * 0.9:
            return {
                'compressed': True,
                'method': self.default_method,
                'data': compressed
            }
        else:
            return {
                'compressed': False,
                'method': None,
                'data': data
            }
    
    def decompress_efficient(self, package: Dict[str, Any]) -> bytes:
        """
        Descomprime dados de um pacote eficiente.
        
        Args:
            package: Pacote retornado por compress_efficient
            
        Returns:
            bytes: Dados descomprimidos
        """
        if not package.get('compressed', False):
            return package['data']
        
        return self.decompress(package['data'], package['method'])