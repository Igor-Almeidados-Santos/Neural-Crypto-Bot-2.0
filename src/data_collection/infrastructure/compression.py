"""
Serviço de compressão de dados para otimização de armazenamento e transmissão.

Este módulo implementa diferentes algoritmos de compressão para dados de mercado,
incluindo compressão especializada para séries temporais e dados estruturados.
"""
import asyncio
import gzip
import lz4.frame
import zstd
import brotli
import pickle
import json
import logging
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any, Union, Tuple, Protocol
from dataclasses import dataclass
from enum import Enum
import io
import struct

import numpy as np
import pandas as pd
from msgpack import packb, unpackb

from data_collection.domain.entities.candle import Candle
from data_collection.domain.entities.orderbook import OrderBook
from data_collection.domain.entities.trade import Trade

logger = logging.getLogger(__name__)


class CompressionAlgorithm(Enum):
    """Algoritmos de compressão disponíveis."""
    GZIP = "gzip"
    LZ4 = "lz4"
    ZSTD = "zstd"
    BROTLI = "brotli"
    MSGPACK = "msgpack"
    CUSTOM_DELTA = "custom_delta"  # Compressão customizada para séries temporais


class SerializationFormat(Enum):
    """Formatos de serialização."""
    JSON = "json"
    PICKLE = "pickle"
    MSGPACK = "msgpack"
    BINARY = "binary"


@dataclass
class CompressionConfig:
    """Configuração para compressão."""
    algorithm: CompressionAlgorithm = CompressionAlgorithm.ZSTD
    level: int = 3  # Nível de compressão (1-22 para zstd, 1-9 para outros)
    serialization: SerializationFormat = SerializationFormat.MSGPACK
    enable_preprocessing: bool = True
    chunk_size: int = 1024 * 1024  # 1MB chunks
    enable_parallel: bool = True
    max_workers: int = 4


@dataclass
class CompressionResult:
    """Resultado da operação de compressão."""
    compressed_data: bytes
    original_size: int
    compressed_size: int
    compression_ratio: float
    algorithm: CompressionAlgorithm
    metadata: Dict[str, Any]
    
    @property
    def space_saved(self) -> float:
        """Percentual de espaço economizado."""
        return (1 - self.compressed_size / self.original_size) * 100


class Compressor(Protocol):
    """Interface para compressores."""
    
    async def compress(self, data: bytes) -> bytes:
        """Comprime dados."""
        ...
    
    async def decompress(self, data: bytes) -> bytes:
        """Descomprime dados."""
        ...


class GzipCompressor:
    """Compressor usando Gzip."""
    
    def __init__(self, level: int = 6):
        self.level = level
    
    async def compress(self, data: bytes) -> bytes:
        return await asyncio.get_event_loop().run_in_executor(
            None, gzip.compress, data, self.level
        )
    
    async def decompress(self, data: bytes) -> bytes:
        return await asyncio.get_event_loop().run_in_executor(
            None, gzip.decompress, data
        )


class LZ4Compressor:
    """Compressor usando LZ4."""
    
    def __init__(self, level: int = 0):
        self.level = level
    
    async def compress(self, data: bytes) -> bytes:
        return await asyncio.get_event_loop().run_in_executor(
            None, lz4.frame.compress, data
        )
    
    async def decompress(self, data: bytes) -> bytes:
        return await asyncio.get_event_loop().run_in_executor(
            None, lz4.frame.decompress, data
        )


class ZstdCompressor:
    """Compressor usando Zstandard."""
    
    def __init__(self, level: int = 3):
        self.level = level
    
    async def compress(self, data: bytes) -> bytes:
        return await asyncio.get_event_loop().run_in_executor(
            None, zstd.compress, data, self.level
        )
    
    async def decompress(self, data: bytes) -> bytes:
        return await asyncio.get_event_loop().run_in_executor(
            None, zstd.decompress, data
        )


class BrotliCompressor:
    """Compressor usando Brotli."""
    
    def __init__(self, level: int = 6):
        self.level = level
    
    async def compress(self, data: bytes) -> bytes:
        return await asyncio.get_event_loop().run_in_executor(
            None, brotli.compress, data, quality=self.level
        )
    
    async def decompress(self, data: bytes) -> bytes:
        return await asyncio.get_event_loop().run_in_executor(
            None, brotli.decompress, data
        )


class DeltaCompressor:
    """
    Compressor customizado para séries temporais usando Delta encoding.
    
    Otimizado para dados de mercado que têm alta correlação temporal.
    """
    
    def __init__(self, base_compressor: Compressor):
        self.base_compressor = base_compressor
    
    async def compress(self, data: bytes) -> bytes:
        """Comprime usando delta encoding + compressor base."""
        try:
            # Deserializa os dados para aplicar delta encoding
            obj = pickle.loads(data)
            
            if isinstance(obj, list) and len(obj) > 1:
                # Aplica delta encoding para listas numéricas
                delta_encoded = self._apply_delta_encoding(obj)
                delta_data = pickle.dumps(delta_encoded)
            else:
                delta_data = data
            
            # Aplica compressor base
            return await self.base_compressor.compress(delta_data)
            
        except Exception as e:
            logger.warning(f"Falha no delta encoding, usando compressor base: {e}")
            return await self.base_compressor.compress(data)
    
    async def decompress(self, data: bytes) -> bytes:
        """Descomprime e reverte delta encoding."""
        # Descomprime usando compressor base
        decompressed = await self.base_compressor.decompress(data)
        
        try:
            obj = pickle.loads(decompressed)
            
            if isinstance(obj, dict) and 'delta_encoded' in obj:
                # Reverte delta encoding
                original = self._revert_delta_encoding(obj)
                return pickle.dumps(original)
            else:
                return decompressed
                
        except Exception as e:
            logger.warning(f"Falha ao reverter delta encoding: {e}")
            return decompressed
    
    def _apply_delta_encoding(self, data_list: List[Any]) -> Dict[str, Any]:
        """Aplica delta encoding em uma lista de dados."""
        if not data_list:
            return {'delta_encoded': True, 'data': []}
        
        # Extrai valores numéricos para delta encoding
        if isinstance(data_list[0], (int, float, Decimal)):
            deltas = [data_list[0]]  # Primeiro valor como referência
            for i in range(1, len(data_list)):
                delta = data_list[i] - data_list[i-1]
                deltas.append(delta)
            
            return {
                'delta_encoded': True,
                'data': deltas,
                'type': 'numeric'
            }
        
        # Para outros tipos, retorna sem modificação
        return {'delta_encoded': False, 'data': data_list}
    
    def _revert_delta_encoding(self, delta_obj: Dict[str, Any]) -> List[Any]:
        """Reverte delta encoding."""
        if not delta_obj.get('delta_encoded'):
            return delta_obj['data']
        
        deltas = delta_obj['data']
        if not deltas:
            return []
        
        if delta_obj.get('type') == 'numeric':
            # Reconstrói valores originais
            original = [deltas[0]]
            for i in range(1, len(deltas)):
                original.append(original[i-1] + deltas[i])
            return original
        
        return deltas


class CompressionService:
    """
    Serviço principal de compressão para dados de mercado.
    
    Implementa compressão inteligente baseada no tipo de dados,
    com suporte a diferentes algoritmos e otimizações específicas.
    """
    
    def __init__(self, config: CompressionConfig = None):
        """
        Inicializa o serviço de compressão.
        
        Args:
            config: Configuração de compressão
        """
        self.config = config or CompressionConfig()
        self._compressors = self._initialize_compressors()
        self._stats = {
            'total_compressed': 0,
            'total_decompressed': 0,
            'bytes_saved': 0,
            'compression_time': 0.0,
            'decompression_time': 0.0
        }
    
    def _initialize_compressors(self) -> Dict[CompressionAlgorithm, Compressor]:
        """Inicializa os compressores disponíveis."""
        compressors = {
            CompressionAlgorithm.GZIP: GzipCompressor(self.config.level),
            CompressionAlgorithm.LZ4: LZ4Compressor(self.config.level),
            CompressionAlgorithm.ZSTD: ZstdCompressor(self.config.level),
            CompressionAlgorithm.BROTLI: BrotliCompressor(self.config.level)
        }
        
        # Adiciona compressor delta se configurado
        if self.config.algorithm == CompressionAlgorithm.CUSTOM_DELTA:
            base_compressor = compressors[CompressionAlgorithm.ZSTD]
            compressors[CompressionAlgorithm.CUSTOM_DELTA] = DeltaCompressor(base_compressor)
        
        return compressors
    
    async def compress_market_data(
        self,
        data: Union[List[Candle], List[Trade], List[OrderBook], Dict[str, Any]],
        algorithm: Optional[CompressionAlgorithm] = None
    ) -> CompressionResult:
        """
        Comprime dados de mercado com otimizações específicas.
        
        Args:
            data: Dados a serem comprimidos
            algorithm: Algoritmo específico (opcional)
            
        Returns:
            CompressionResult: Resultado da compressão
        """
        start_time = asyncio.get_event_loop().time()
        algorithm = algorithm or self.config.algorithm
        
        try:
            # Preprocessa os dados se habilitado
            if self.config.enable_preprocessing:
                preprocessed_data = self._preprocess_market_data(data)
            else:
                preprocessed_data = data
            
            # Serializa os dados
            serialized_data = await self._serialize_data(preprocessed_data)
            original_size = len(serialized_data)
            
            # Comprime os dados
            compressor = self._compressors[algorithm]
            compressed_data = await compressor.compress(serialized_data)
            compressed_size = len(compressed_data)
            
            # Calcula métricas
            compression_ratio = compressed_size / original_size
            compression_time = asyncio.get_event_loop().time() - start_time
            
            # Atualiza estatísticas
            self._stats['total_compressed'] += 1
            self._stats['bytes_saved'] += (original_size - compressed_size)
            self._stats['compression_time'] += compression_time
            
            # Cria resultado
            result = CompressionResult(
                compressed_data=compressed_data,
                original_size=original_size,
                compressed_size=compressed_size,
                compression_ratio=compression_ratio,
                algorithm=algorithm,
                metadata={
                    'serialization': self.config.serialization.value,
                    'preprocessing': self.config.enable_preprocessing,
                    'compression_time': compression_time,
                    'data_type': type(data).__name__
                }
            )
            
            logger.debug(
                f"Dados comprimidos: {original_size} -> {compressed_size} bytes "
                f"({result.space_saved:.1f}% economizado) em {compression_time:.3f}s"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Erro na compressão: {str(e)}")
            raise
    
    async def decompress_market_data(
        self,
        compressed_result: CompressionResult
    ) -> Union[List[Candle], List[Trade], List[OrderBook], Dict[str, Any]]:
        """
        Descomprime dados de mercado.
        
        Args:
            compressed_result: Resultado da compressão original
            
        Returns:
            Dados descomprimidos originais
        """
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Descomprime os dados
            compressor = self._compressors[compressed_result.algorithm]
            decompressed_data = await compressor.decompress(compressed_result.compressed_data)
            
            # Deserializa os dados
            serialization_format = SerializationFormat(
                compressed_result.metadata.get('serialization', 'msgpack')
            )
            deserialized_data = await self._deserialize_data(decompressed_data, serialization_format)
            
            # Reverte preprocessamento se necessário
            if compressed_result.metadata.get('preprocessing', False):
                final_data = self._postprocess_market_data(deserialized_data)
            else:
                final_data = deserialized_data
            
            # Atualiza estatísticas
            decompression_time = asyncio.get_event_loop().time() - start_time
            self._stats['total_decompressed'] += 1
            self._stats['decompression_time'] += decompression_time
            
            logger.debug(f"Dados descomprimidos em {decompression_time:.3f}s")
            
            return final_data
            
        except Exception as e:
            logger.error(f"Erro na descompressão: {str(e)}")
            raise
    
    async def compress_bulk_data(
        self,
        data_chunks: List[Any],
        algorithm: Optional[CompressionAlgorithm] = None
    ) -> List[CompressionResult]:
        """
        Comprime múltiplos chunks de dados em paralelo.
        
        Args:
            data_chunks: Lista de chunks para comprimir
            algorithm: Algoritmo de compressão
            
        Returns:
            Lista de resultados de compressão
        """
        if not self.config.enable_parallel or len(data_chunks) == 1:
            # Processamento sequencial
            results = []
            for chunk in data_chunks:
                result = await self.compress_market_data(chunk, algorithm)
                results.append(result)
            return results
        
        # Processamento paralelo
        tasks = [
            self.compress_market_data(chunk, algorithm)
            for chunk in data_chunks
        ]
        
        # Limita o número de tarefas concorrentes
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def compress_with_semaphore(task):
            async with semaphore:
                return await task
        
        wrapped_tasks = [compress_with_semaphore(task) for task in tasks]
        results = await asyncio.gather(*wrapped_tasks)
        
        return results
    
    def _preprocess_market_data(self, data: Any) -> Any:
        """
        Preprocessa dados de mercado para melhor compressão.
        
        Args:
            data: Dados originais
            
        Returns:
            Dados preprocessados
        """
        try:
            if isinstance(data, list):
                if not data:
                    return data
                
                # Detecta tipo de dados
                sample = data[0]
                
                if isinstance(sample, Candle):
                    return self._preprocess_candles(data)
                elif isinstance(sample, Trade):
                    return self._preprocess_trades(data)
                elif isinstance(sample, OrderBook):
                    return self._preprocess_orderbooks(data)
            
            return data
            
        except Exception as e:
            logger.warning(f"Erro no preprocessamento: {e}")
            return data
    
    def _preprocess_candles(self, candles: List[Candle]) -> Dict[str, Any]:
        """Preprocessa lista de candles para compressão otimizada."""
        if not candles:
            return {'type': 'candles', 'data': []}
        
        # Agrupa por exchange e trading_pair para melhor compressão
        grouped = {}
        for candle in candles:
            key = f"{candle.exchange}:{candle.trading_pair}:{candle.timeframe.value}"
            if key not in grouped:
                grouped[key] = {
                    'exchange': candle.exchange,
                    'trading_pair': candle.trading_pair,
                    'timeframe': candle.timeframe.value,
                    'timestamps': [],
                    'opens': [],
                    'highs': [],
                    'lows': [],
                    'closes': [],
                    'volumes': [],
                    'trades': []
                }
            
            group = grouped[key]
            group['timestamps'].append(candle.timestamp.timestamp())
            group['opens'].append(float(candle.open))
            group['highs'].append(float(candle.high))
            group['lows'].append(float(candle.low))
            group['closes'].append(float(candle.close))
            group['volumes'].append(float(candle.volume))
            group['trades'].append(candle.trades)
        
        return {
            'type': 'candles',
            'data': grouped
        }
    
    def _preprocess_trades(self, trades: List[Trade]) -> Dict[str, Any]:
        """Preprocessa lista de trades para compressão otimizada."""
        if not trades:
            return {'type': 'trades', 'data': []}
        
        # Agrupa por exchange e trading_pair
        grouped = {}
        for trade in trades:
            key = f"{trade.exchange}:{trade.trading_pair}"
            if key not in grouped:
                grouped[key] = {
                    'exchange': trade.exchange,
                    'trading_pair': trade.trading_pair,
                    'ids': [],
                    'timestamps': [],
                    'prices': [],
                    'amounts': [],
                    'sides': [],
                    'takers': []
                }
            
            group = grouped[key]
            group['ids'].append(trade.id)
            group['timestamps'].append(trade.timestamp.timestamp())
            group['prices'].append(float(trade.price))
            group['amounts'].append(float(trade.amount))
            group['sides'].append(trade.side.value)
            group['takers'].append(trade.taker)
        
        return {
            'type': 'trades',
            'data': grouped
        }
    
    def _preprocess_orderbooks(self, orderbooks: List[OrderBook]) -> Dict[str, Any]:
        """Preprocessa lista de orderbooks para compressão otimizada."""
        # Para orderbooks, mantém estrutura original por ser mais complexa
        return {
            'type': 'orderbooks',
            'data': [ob.__dict__ for ob in orderbooks]
        }
    
    def _postprocess_market_data(self, data: Dict[str, Any]) -> Any:
        """Reverte o preprocessamento dos dados."""
        try:
            data_type = data.get('type')
            
            if data_type == 'candles':
                return self._postprocess_candles(data['data'])
            elif data_type == 'trades':
                return self._postprocess_trades(data['data'])
            elif data_type == 'orderbooks':
                return self._postprocess_orderbooks(data['data'])
            
            return data
            
        except Exception as e:
            logger.warning(f"Erro no pós-processamento: {e}")
            return data
    
    def _postprocess_candles(self, grouped_data: Dict[str, Any]) -> List[Candle]:
        """Reconstrói candles a partir de dados agrupados."""
        candles = []
        
        for key, group in grouped_data.items():
            for i in range(len(group['timestamps'])):
                candle = Candle(
                    exchange=group['exchange'],
                    trading_pair=group['trading_pair'],
                    timestamp=datetime.fromtimestamp(group['timestamps'][i]),
                    timeframe=group['timeframe'],
                    open=Decimal(str(group['opens'][i])),
                    high=Decimal(str(group['highs'][i])),
                    low=Decimal(str(group['lows'][i])),
                    close=Decimal(str(group['closes'][i])),
                    volume=Decimal(str(group['volumes'][i])),
                    trades=group['trades'][i]
                )
                candles.append(candle)
        
        return candles
    
    def _postprocess_trades(self, grouped_data: Dict[str, Any]) -> List[Trade]:
        """Reconstrói trades a partir de dados agrupados."""
        trades = []
        
        for key, group in grouped_data.items():
            for i in range(len(group['timestamps'])):
                trade = Trade(
                    id=group['ids'][i],
                    exchange=group['exchange'],
                    trading_pair=group['trading_pair'],
                    timestamp=datetime.fromtimestamp(group['timestamps'][i]),
                    price=Decimal(str(group['prices'][i])),
                    amount=Decimal(str(group['amounts'][i])),
                    cost=Decimal(str(group['prices'][i])) * Decimal(str(group['amounts'][i])),
                    side=group['sides'][i],
                    taker=group['takers'][i]
                )
                trades.append(trade)
        
        return trades
    
    def _postprocess_orderbooks(self, data: List[Dict[str, Any]]) -> List[OrderBook]:
        """Reconstrói orderbooks a partir de dados serializados."""
        # Implementação seria mais complexa, requerendo reconstrução das entidades
        # Por simplicidade, retorna dados como estão
        return data
    
    async def _serialize_data(self, data: Any) -> bytes:
        """Serializa dados baseado na configuração."""
        if self.config.serialization == SerializationFormat.JSON:
            return json.dumps(data, default=str).encode('utf-8')
        elif self.config.serialization == SerializationFormat.PICKLE:
            return pickle.dumps(data)
        elif self.config.serialization == SerializationFormat.MSGPACK:
            return packb(data, default=str)
        else:
            raise ValueError(f"Formato de serialização não suportado: {self.config.serialization}")
    
    async def _deserialize_data(self, data: bytes, format: SerializationFormat) -> Any:
        """Deserializa dados baseado no formato."""
        if format == SerializationFormat.JSON:
            return json.loads(data.decode('utf-8'))
        elif format == SerializationFormat.PICKLE:
            return pickle.loads(data)
        elif format == SerializationFormat.MSGPACK:
            return unpackb(data, raw=False)
        else:
            raise ValueError(f"Formato de deserialização não suportado: {format}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do serviço de compressão."""
        stats = self._stats.copy()
        
        if stats['total_compressed'] > 0:
            stats['avg_compression_time'] = stats['compression_time'] / stats['total_compressed']
        
        if stats['total_decompressed'] > 0:
            stats['avg_decompression_time'] = stats['decompression_time'] / stats['total_decompressed']
        
        return stats
    
    def reset_stats(self) -> None:
        """Reseta as estatísticas."""
        self._stats = {
            'total_compressed': 0,
            'total_decompressed': 0,
            'bytes_saved': 0,
            'compression_time': 0.0,
            'decompression_time': 0.0
        }


# Funções de conveniência para uso direto
async def compress_candles(
    candles: List[Candle],
    algorithm: CompressionAlgorithm = CompressionAlgorithm.ZSTD
) -> CompressionResult:
    """Comprime uma lista de candles."""
    service = CompressionService()
    return await service.compress_market_data(candles, algorithm)


async def compress_trades(
    trades: List[Trade],
    algorithm: CompressionAlgorithm = CompressionAlgorithm.ZSTD
) -> CompressionResult:
    """Comprime uma lista de trades."""
    service = CompressionService()
    return await service.compress_market_data(trades, algorithm)


async def compress_orderbooks(
    orderbooks: List[OrderBook],
    algorithm: CompressionAlgorithm = CompressionAlgorithm.ZSTD
) -> CompressionResult:
    """Comprime uma lista de orderbooks."""
    service = CompressionService()
    return await service.compress_market_data(orderbooks, algorithm)