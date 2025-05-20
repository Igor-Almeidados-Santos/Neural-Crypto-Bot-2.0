"""
Neural Crypto Bot - Authentication Middleware

Este módulo implementa o middleware de autenticação para a API.
Responsável por validar os tokens JWT e fornecer informações do usuário para os endpoints protegidos.
"""
import logging
import time
from typing import Callable, Dict, Optional, Union

from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from jose import JWTError, jwt
from starlette.middleware.base import BaseHTTPMiddleware

from src.common.utils.config import get_settings

logger = logging.getLogger(__name__)


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Middleware para autenticação de requisições usando tokens JWT.
    
    Esta classe verifica a presença e validade de tokens JWT nas requisições,
    permitindo ou negando acesso aos recursos protegidos da API.
    """

    def __init__(
        self,
        app,
        exclude_paths: Optional[list] = None,
    ):
        """
        Inicializa o middleware de autenticação.
        
        Args:
            app: A aplicação FastAPI.
            exclude_paths: Lista de caminhos (endpoints) que não requerem autenticação.
        """
        super().__init__(app)
        self.exclude_paths = exclude_paths or [
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/auth/login",
            "/api/v1/auth/refresh",
        ]
        self.settings = get_settings()

    async def dispatch(
        self, request: Request, call_next: Callable
    ) -> Union[Response, JSONResponse]:
        """
        Processa a requisição e verifica a autenticação quando necessário.
        
        Args:
            request: Objeto Request contendo dados da requisição.
            call_next: Função para processar o próximo middleware/endpoint.
            
        Returns:
            Response ou JSONResponse dependendo do resultado da validação.
        """
        # Ignora autenticação para caminhos excluídos
        if self._is_path_excluded(request.url.path):
            return await call_next(request)

        # Extrai e valida o token de autorização
        token = self._extract_token(request)
        if not token:
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Token de autenticação não fornecido"},
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Valida o token JWT
        try:
            user_data = self._validate_token(token)
            if not user_data:
                return JSONResponse(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    content={"detail": "Token de autenticação inválido ou expirado"},
                    headers={"WWW-Authenticate": "Bearer"},
                )

            # Adiciona informações do usuário à request para uso nos endpoints
            request.state.user = user_data
            
            # Adiciona log para fins de auditoria
            logger.info(
                f"Acesso autenticado: user_id={user_data.get('sub')}, "
                f"endpoint={request.url.path}, method={request.method}"
            )
            
            return await call_next(request)
        
        except JWTError as e:
            logger.warning(f"Erro na validação do token JWT: {str(e)}")
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={"detail": "Token de autenticação inválido"},
                headers={"WWW-Authenticate": "Bearer"},
            )
        except Exception as e:
            logger.error(f"Erro inesperado na autenticação: {str(e)}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Erro interno no servidor de autenticação"},
            )

    def _is_path_excluded(self, path: str) -> bool:
        """
        Verifica se o caminho está excluído da autenticação.
        
        Args:
            path: Caminho da URL a ser verificado.
            
        Returns:
            True se o caminho estiver excluído, False caso contrário.
        """
        return any(path.startswith(excluded) for excluded in self.exclude_paths)

    def _extract_token(self, request: Request) -> Optional[str]:
        """
        Extrai o token JWT do cabeçalho de autorização.
        
        Args:
            request: Objeto Request contendo dados da requisição.
            
        Returns:
            Token JWT ou None se não for encontrado.
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return None

        parts = auth_header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return None

        return parts[1]

    def _validate_token(self, token: str) -> Optional[Dict]:
        """
        Valida o token JWT e retorna as informações do usuário.
        
        Args:
            token: Token JWT a ser validado.
            
        Returns:
            Dicionário contendo informações do usuário ou None se inválido.
            
        Raises:
            JWTError: Se houver erro na decodificação do token.
        """
        try:
            # Decodifica o token JWT usando a chave secreta e algoritmo configurados
            payload = jwt.decode(
                token,
                self.settings.SECRET_KEY,
                algorithms=[self.settings.JWT_ALGORITHM],
            )
            
            # Verifica se o token está expirado
            exp_timestamp = payload.get("exp")
            if exp_timestamp is None:
                logger.warning("Token sem campo 'exp' (expiração)")
                return None
                
            if time.time() > exp_timestamp:
                logger.warning("Token expirado")
                return None
                
            return payload
        
        except JWTError as e:
            logger.warning(f"Erro ao decodificar token: {str(e)}")
            raise