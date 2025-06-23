#!/usr/bin/env python3
"""
Script de validação para verificar se todas as correções da Fase 1 foram implementadas.

Este script verifica:
- Arquivos de infraestrutura missing foram criados
- Imports estão funcionando
- Configurações estão corretas
- Dependências foram atualizadas
- Sistema está funcional
"""
import sys
import os
import asyncio
import importlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import json
import traceback
from datetime import datetime


class Colors:
    """Cores para output do terminal."""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'


class ValidationResult:
    """Resultado de uma validação."""
    
    def __init__(self, name: str, passed: bool, message: str, details: Optional[Dict] = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.utcnow()


class Phase1Validator:
    """Validador para as correções da Fase 1."""
    
    def __init__(self):
        """Inicializa o validador."""
        self.project_root = Path(__file__).parent.parent
        self.results: List[ValidationResult] = []
        self.errors: List[str] = []
        
        # Adicionar src ao path para imports
        sys.path.insert(0, str(self.project_root / "src"))
    
    def print_header(self):
        """Imprime cabeçalho do validador."""
        print(f"{Colors.BOLD}{Colors.PURPLE}")
        print("=" * 80)
        print("         VALIDADOR FASE 1 - CORREÇÕES CRÍTICAS")
        print("              Neural Crypto Bot v2.0")
        print("=" * 80)
        print(f"{Colors.END}\n")
    
    def print_section(self, title: str):
        """Imprime título de seção."""
        print(f"{Colors.BOLD}{Colors.CYAN}🔍 {title}{Colors.END}")
        print("-" * 50)
    
    def print_result(self, result: ValidationResult):
        """Imprime resultado de uma validação."""
        status_color = Colors.GREEN if result.passed else Colors.RED
        status_symbol = "✅" if result.passed else "❌"
        
        print(f"{status_color}{status_symbol} {result.name}{Colors.END}")
        print(f"   {result.message}")
        
        if result.details and not result.passed:
            for key, value in result.details.items():
                print(f"   📋 {key}: {value}")
        print()
    
    def add_result(self, name: str, passed: bool, message: str, details: Optional[Dict] = None):
        """Adiciona resultado de validação."""
        result = ValidationResult(name, passed, message, details)
        self.results.append(result)
        self.print_result(result)
        return result
    
    # Validações de arquivos
    
    def validate_missing_files(self) -> bool:
        """Valida se os arquivos missing foram criados."""
        self.print_section("VALIDAÇÃO DE ARQUIVOS MISSING")
        
        required_files = [
            "src/data_collection/infrastructure/crypto.py",
            "src/data_collection/infrastructure/compression.py", 
            "src/data_collection/infrastructure/load_balancer.py",
            "src/data_collection/infrastructure/health_check.py"
        ]
        
        all_files_exist = True
        missing_files = []
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            
            if full_path.exists():
                self.add_result(
                    f"Arquivo {file_path}",
                    True,
                    f"Arquivo existe ({full_path.stat().st_size} bytes)"
                )
            else:
                missing_files.append(file_path)
                all_files_exist = False
                self.add_result(
                    f"Arquivo {file_path}",
                    False,
                    "Arquivo não encontrado"
                )
        
        if missing_files:
            self.add_result(
                "Arquivos Missing - Resumo",
                False,
                f"{len(missing_files)} arquivos não encontrados",
                {"missing_files": missing_files}
            )
        else:
            self.add_result(
                "Arquivos Missing - Resumo",
                True,
                "Todos os arquivos foram criados com sucesso"
            )
        
        return all_files_exist
    
    def validate_file_structure(self) -> bool:
        """Valida estrutura de arquivos do módulo."""
        self.print_section("VALIDAÇÃO DE ESTRUTURA")
        
        expected_structure = {
            "src/data_collection": "directory",
            "src/data_collection/main.py": "file",
            "src/data_collection/main_extended.py": "file", 
            "src/data_collection/adapters": "directory",
            "src/data_collection/adapters/exchange_adapter_interface.py": "file",
            "src/data_collection/application": "directory",
            "src/data_collection/application/services": "directory",
            "src/data_collection/domain": "directory",
            "src/data_collection/domain/entities": "directory",
            "src/data_collection/infrastructure": "directory"
        }
        
        structure_valid = True
        
        for path, expected_type in expected_structure.items():
            full_path = self.project_root / path
            
            if not full_path.exists():
                structure_valid = False
                self.add_result(
                    f"Estrutura - {path}",
                    False,
                    f"{expected_type.title()} não encontrado"
                )
            elif expected_type == "directory" and not full_path.is_dir():
                structure_valid = False
                self.add_result(
                    f"Estrutura - {path}",
                    False,
                    "Deveria ser diretório mas é arquivo"
                )
            elif expected_type == "file" and not full_path.is_file():
                structure_valid = False
                self.add_result(
                    f"Estrutura - {path}",
                    False,
                    "Deveria ser arquivo mas é diretório"
                )
            else:
                self.add_result(
                    f"Estrutura - {path}",
                    True,
                    f"{expected_type.title()} OK"
                )
        
        return structure_valid
    
    # Validações de imports
    
    def validate_imports(self) -> bool:
        """Valida se todos os imports estão funcionando."""
        self.print_section("VALIDAÇÃO DE IMPORTS")
        
        imports_to_test = [
            ("crypto", "src.data_collection.infrastructure.crypto", "CryptoService"),
            ("compression", "src.data_collection.infrastructure.compression", "CompressionService"),
            ("load_balancer", "src.data_collection.infrastructure.load_balancer", "ExchangeLoadBalancer"),
            ("health_check", "src.data_collection.infrastructure.health_check", "HealthCheckService"),
            ("main_extended", "src.data_collection.main_extended", "ExtendedDataCollectionService"),
            ("exchange_interface", "src.data_collection.adapters.exchange_adapter_interface", "ExchangeAdapterInterface")
        ]
        
        all_imports_work = True
        
        for name, module_path, class_name in imports_to_test:
            try:
                module = importlib.import_module(module_path)
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    self.add_result(
                        f"Import {name}",
                        True,
                        f"✓ {module_path}.{class_name} importado com sucesso"
                    )
                else:
                    all_imports_work = False
                    self.add_result(
                        f"Import {name}",
                        False,
                        f"Classe {class_name} não encontrada em {module_path}"
                    )
            except ImportError as e:
                all_imports_work = False
                self.add_result(
                    f"Import {name}",
                    False,
                    f"Erro de import: {str(e)}"
                )
            except Exception as e:
                all_imports_work = False
                self.add_result(
                    f"Import {name}",
                    False,
                    f"Erro inesperado: {str(e)}"
                )
        
        return all_imports_work
    
    def validate_dependencies(self) -> bool:
        """Valida dependências do pyproject.toml."""
        self.print_section("VALIDAÇÃO DE DEPENDÊNCIAS")
        
        pyproject_path = self.project_root / "pyproject.toml"
        
        if not pyproject_path.exists():
            self.add_result(
                "pyproject.toml",
                False,
                "Arquivo pyproject.toml não encontrado"
            )
            return False
        
        try:
            # Verificar se Poetry consegue ler o arquivo
            result = subprocess.run(
                ["poetry", "check"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                self.add_result(
                    "Poetry Check",
                    True,
                    "pyproject.toml válido"
                )
            else:
                self.add_result(
                    "Poetry Check", 
                    False,
                    f"Erro na validação: {result.stderr}",
                    {"stdout": result.stdout, "stderr": result.stderr}
                )
                return False
                
        except subprocess.TimeoutExpired:
            self.add_result(
                "Poetry Check",
                False,
                "Timeout na validação do Poetry"
            )
            return False
        except FileNotFoundError:
            self.add_result(
                "Poetry Check",
                False,
                "Poetry não encontrado no sistema"
            )
            return False
        except Exception as e:
            self.add_result(
                "Poetry Check",
                False,
                f"Erro inesperado: {str(e)}"
            )
            return False
        
        # Verificar dependências críticas
        critical_deps = [
            "ccxt", "asyncio", "websockets", "structlog", "confluent-kafka",
            "sqlalchemy", "prometheus-client", "cryptography", "psutil"
        ]
        
        try:
            result = subprocess.run(
                ["poetry", "show"],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                installed_packages = result.stdout.lower()
                missing_deps = []
                
                for dep in critical_deps:
                    if dep in installed_packages:
                        self.add_result(
                            f"Dependência {dep}",
                            True,
                            "Instalada"
                        )
                    else:
                        missing_deps.append(dep)
                        self.add_result(
                            f"Dependência {dep}",
                            False,
                            "Não instalada"
                        )
                
                if missing_deps:
                    self.add_result(
                        "Dependências Críticas",
                        False,
                        f"{len(missing_deps)} dependências missing",
                        {"missing": missing_deps}
                    )
                    return False
                else:
                    self.add_result(
                        "Dependências Críticas",
                        True,
                        "Todas as dependências críticas instaladas"
                    )
            
        except Exception as e:
            self.add_result(
                "Lista de Dependências",
                False,
                f"Erro ao listar dependências: {str(e)}"
            )
            return False
        
        return True
    
    # Validações funcionais
    
    async def validate_service_initialization(self) -> bool:
        """Valida se o serviço pode ser inicializado."""
        self.print_section("VALIDAÇÃO DE INICIALIZAÇÃO")
        
        try:
            # Import do serviço
            from src.data_collection.main_extended import ExtendedDataCollectionService
            
            # Configuração mínima para teste
            test_config = {
                'collection': {
                    'pairs': ['BTC/USDT'],
                    'exchanges': ['binance'],
                    'timeframes': ['1m'],
                    'interval_seconds': 60
                },
                'exchanges': {
                    'binance': {
                        'testnet': True,
                        'api_key': None,
                        'api_secret': None
                    }
                },
                'storage': {
                    'enable_database': False,
                    'enable_kafka': False
                },
                'load_balancing': {
                    'enabled': True,
                    'strategy': 'round_robin'
                }
            }
            
            # Tentar criar instância
            service = ExtendedDataCollectionService(test_config)
            
            self.add_result(
                "Criação do Serviço",
                True,
                "ExtendedDataCollectionService criado com sucesso"
            )
            
            # Verificar atributos essenciais
            required_attrs = [
                'crypto_service', 'compression_service', 'exchange_load_balancer',
                'normalization_service', 'validation_service', 'performance_monitor',
                'health_checker'
            ]
            
            missing_attrs = []
            for attr in required_attrs:
                if hasattr(service, attr):
                    self.add_result(
                        f"Atributo {attr}",
                        True,
                        "Presente"
                    )
                else:
                    missing_attrs.append(attr)
                    self.add_result(
                        f"Atributo {attr}",
                        False,
                        "Ausente"
                    )
            
            if missing_attrs:
                self.add_result(
                    "Atributos do Serviço",
                    False,
                    f"{len(missing_attrs)} atributos ausentes",
                    {"missing": missing_attrs}
                )
                return False
            
            self.add_result(
                "Inicialização do Serviço",
                True,
                "Serviço pode ser criado com configuração de teste"
            )
            
            return True
            
        except Exception as e:
            self.add_result(
                "Inicialização do Serviço",
                False,
                f"Erro na inicialização: {str(e)}",
                {"traceback": traceback.format_exc()}
            )
            return False
    
    async def validate_health_check_system(self) -> bool:
        """Valida sistema de health check."""
        self.print_section("VALIDAÇÃO DE HEALTH CHECK")
        
        try:
            from src.data_collection.infrastructure.health_check import HealthCheckService, quick_health_check
            
            # Testar criação do serviço
            health_service = HealthCheckService()
            
            self.add_result(
                "HealthCheckService",
                True,
                "Serviço de health check criado"
            )
            
            # Testar quick health check
            result = await quick_health_check()
            
            if isinstance(result, dict) and 'healthy' in result:
                self.add_result(
                    "Quick Health Check",
                    True,
                    f"Health check executado: {result['status']}"
                )
            else:
                self.add_result(
                    "Quick Health Check",
                    False,
                    "Formato de resposta inválido"
                )
                return False
            
            # Testar execução de todos os checks
            report = await health_service.run_all_checks()
            
            if hasattr(report, 'overall_status') and hasattr(report, 'checks'):
                self.add_result(
                    "Health Check Completo",
                    True,
                    f"Relatório gerado com {len(report.checks)} verificações"
                )
            else:
                self.add_result(
                    "Health Check Completo",
                    False,
                    "Formato de relatório inválido"
                )
                return False
            
            return True
            
        except Exception as e:
            self.add_result(
                "Sistema de Health Check",
                False,
                f"Erro: {str(e)}",
                {"traceback": traceback.format_exc()}
            )
            return False
    
    def validate_crypto_service(self) -> bool:
        """Valida serviço de criptografia."""
        self.print_section("VALIDAÇÃO DE CRIPTOGRAFIA")
        
        try:
            from src.data_collection.infrastructure.crypto import CryptoService
            
            # Criar serviço
            crypto = CryptoService()
            
            # Testar criptografia básica
            test_data = "test_data_for_encryption"
            encrypted = crypto.encrypt(test_data)
            decrypted = crypto.decrypt(encrypted)
            
            if decrypted == test_data:
                self.add_result(
                    "Criptografia Básica",
                    True,
                    "Encrypt/decrypt funcionando"
                )
            else:
                self.add_result(
                    "Criptografia Básica",
                    False,
                    "Dados descriptografados não coincidem"
                )
                return False
            
            # Testar criptografia de dicionário
            test_dict = {
                'api_key': 'secret_key_123',
                'normal_field': 'normal_value'
            }
            
            encrypted_dict = crypto.encrypt_dict(test_dict)
            decrypted_dict = crypto.decrypt_dict(encrypted_dict)
            
            if decrypted_dict['api_key'] == test_dict['api_key']:
                self.add_result(
                    "Criptografia de Dicionário",
                    True,
                    "Encrypt/decrypt de dict funcionando"
                )
            else:
                self.add_result(
                    "Criptografia de Dicionário",
                    False,
                    "Campos não descriptografados corretamente"
                )
                return False
            
            # Testar health check
            health = crypto.health_check()
            if health.get('healthy', False):
                self.add_result(
                    "Health Check Crypto",
                    True,
                    "Serviço de criptografia saudável"
                )
            else:
                self.add_result(
                    "Health Check Crypto",
                    False,
                    "Health check falhou"
                )
                return False
            
            return True
            
        except Exception as e:
            self.add_result(
                "Serviço de Criptografia",
                False,
                f"Erro: {str(e)}",
                {"traceback": traceback.format_exc()}
            )
            return False
    
    def validate_compression_service(self) -> bool:
        """Valida serviço de compressão."""
        self.print_section("VALIDAÇÃO DE COMPRESSÃO")
        
        try:
            from src.data_collection.infrastructure.compression import CompressionService
            
            # Criar serviço
            compression = CompressionService()
            
            # Testar compressão básica
            test_data = "test data for compression " * 100  # Dados repetitivos comprimem bem
            
            result = compression.compress(test_data)
            
            if result.compressed_size < result.original_size:
                self.add_result(
                    "Compressão Básica",
                    True,
                    f"Compressão OK: {result.original_size} -> {result.compressed_size} bytes (ratio: {result.compression_ratio:.2f})"
                )
            else:
                self.add_result(
                    "Compressão Básica",
                    False,
                    "Dados não foram comprimidos efetivamente"
                )
                return False
            
            # Testar descompressão
            decompressed = compression.decompress(result.compressed_data, result.method)
            
            if decompressed.decode('utf-8') == test_data:
                self.add_result(
                    "Descompressão",
                    True,
                    "Dados descomprimidos corretamente"
                )
            else:
                self.add_result(
                    "Descompressão",
                    False,
                    "Dados descomprimidos não coincidem"
                )
                return False
            
            # Testar métodos suportados
            supported_methods = compression.get_supported_methods()
            
            if len(supported_methods) >= 3:  # Pelo menos zlib, gzip, bz2
                self.add_result(
                    "Métodos de Compressão",
                    True,
                    f"{len(supported_methods)} métodos suportados"
                )
            else:
                self.add_result(
                    "Métodos de Compressão",
                    False,
                    f"Apenas {len(supported_methods)} métodos suportados"
                )
            
            # Testar health check
            health = compression.health_check()
            if health.get('healthy', False):
                self.add_result(
                    "Health Check Compression",
                    True,
                    "Serviço de compressão saudável"
                )
            else:
                self.add_result(
                    "Health Check Compression", 
                    False,
                    "Health check falhou"
                )
                return False
            
            return True
            
        except Exception as e:
            self.add_result(
                "Serviço de Compressão",
                False,
                f"Erro: {str(e)}",
                {"traceback": traceback.format_exc()}
            )
            return False
    
    def validate_load_balancer(self) -> bool:
        """Valida load balancer."""
        self.print_section("VALIDAÇÃO DE LOAD BALANCER")
        
        try:
            from src.data_collection.infrastructure.load_balancer import ExchangeLoadBalancer, BalancingStrategy
            
            # Criar load balancer
            lb = ExchangeLoadBalancer()
            
            self.add_result(
                "Criação Load Balancer",
                True,
                "Load balancer criado com sucesso"
            )
            
            # Testar estratégias
            strategies = list(BalancingStrategy)
            
            if len(strategies) >= 5:  # Pelo menos 5 estratégias
                self.add_result(
                    "Estratégias de Balanceamento",
                    True,
                    f"{len(strategies)} estratégias disponíveis"
                )
            else:
                self.add_result(
                    "Estratégias de Balanceamento",
                    False,
                    f"Apenas {len(strategies)} estratégias disponíveis"
                )
            
            # Testar health check
            health = lb.health_check()
            if isinstance(health, dict):
                self.add_result(
                    "Health Check Load Balancer",
                    True,
                    "Health check retorna formato correto"
                )
            else:
                self.add_result(
                    "Health Check Load Balancer",
                    False,
                    "Health check retorna formato incorreto"
                )
                return False
            
            return True
            
        except Exception as e:
            self.add_result(
                "Load Balancer",
                False,
                f"Erro: {str(e)}",
                {"traceback": traceback.format_exc()}
            )
            return False
    
    # Relatório final
    
    def generate_report(self) -> Dict[str, Any]:
        """Gera relatório final de validação."""
        total_checks = len(self.results)
        passed_checks = sum(1 for r in self.results if r.passed)
        failed_checks = total_checks - passed_checks
        
        success_rate = (passed_checks / total_checks * 100) if total_checks > 0 else 0
        
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'summary': {
                'total_checks': total_checks,
                'passed': passed_checks,
                'failed': failed_checks,
                'success_rate': success_rate
            },
            'overall_result': success_rate >= 90,  # 90% success rate mínimo
            'results': [
                {
                    'name': r.name,
                    'passed': r.passed,
                    'message': r.message,
                    'details': r.details
                }
                for r in self.results
            ],
            'errors': self.errors
        }
    
    def print_summary(self, report: Dict[str, Any]):
        """Imprime resumo final."""
        print("\n" + "=" * 80)
        print(f"{Colors.BOLD}{Colors.WHITE}RESUMO FINAL DA VALIDAÇÃO{Colors.END}")
        print("=" * 80)
        
        summary = report['summary']
        
        # Status geral
        if report['overall_result']:
            status_color = Colors.GREEN
            status_text = "✅ APROVADO"
        else:
            status_color = Colors.RED
            status_text = "❌ REPROVADO"
        
        print(f"\n{Colors.BOLD}Status Geral: {status_color}{status_text}{Colors.END}")
        print(f"\n📊 Estatísticas:")
        print(f"   Total de verificações: {summary['total_checks']}")
        print(f"   {Colors.GREEN}✅ Aprovadas: {summary['passed']}{Colors.END}")
        print(f"   {Colors.RED}❌ Reprovadas: {summary['failed']}{Colors.END}")
        print(f"   Taxa de sucesso: {summary['success_rate']:.1f}%")
        
        if summary['failed'] > 0:
            print(f"\n{Colors.RED}🚨 Verificações que falharam:{Colors.END}")
            for result in self.results:
                if not result.passed:
                    print(f"   ❌ {result.name}: {result.message}")
        
        print(f"\n{Colors.CYAN}📅 Validação executada em: {report['timestamp']}{Colors.END}")
        print("=" * 80)
    
    async def run_all_validations(self) -> bool:
        """Executa todas as validações."""
        self.print_header()
        
        validations = [
            ("Arquivos Missing", self.validate_missing_files),
            ("Estrutura", self.validate_file_structure),
            ("Imports", self.validate_imports),
            ("Dependências", self.validate_dependencies),
            ("Inicialização", self.validate_service_initialization),
            ("Health Check", self.validate_health_check_system),
            ("Criptografia", self.validate_crypto_service),
            ("Compressão", self.validate_compression_service),
            ("Load Balancer", self.validate_load_balancer)
        ]
        
        overall_success = True
        
        for name, validation_func in validations:
            try:
                if asyncio.iscoroutinefunction(validation_func):
                    success = await validation_func()
                else:
                    success = validation_func()
                
                if not success:
                    overall_success = False
                    
            except Exception as e:
                self.errors.append(f"Erro na validação {name}: {str(e)}")
                overall_success = False
                self.add_result(
                    f"Validação {name}",
                    False,
                    f"Erro inesperado: {str(e)}"
                )
        
        # Gerar e exibir relatório
        report = self.generate_report()
        self.print_summary(report)
        
        # Salvar relatório
        report_path = self.project_root / "validation_report.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📄 Relatório salvo em: {report_path}")
        
        return overall_success and report['overall_result']


async def main():
    """Função principal."""
    validator = Phase1Validator()
    
    try:
        success = await validator.run_all_validations()
        
        if success:
            print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 FASE 1 VALIDADA COM SUCESSO!{Colors.END}")
            print(f"{Colors.GREEN}Todas as correções críticas foram implementadas corretamente.{Colors.END}")
            return 0
        else:
            print(f"\n{Colors.RED}{Colors.BOLD}💥 FASE 1 COM PROBLEMAS!{Colors.END}")
            print(f"{Colors.RED}Algumas correções precisam ser revisadas.{Colors.END}")
            return 1
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}⚠️ Validação interrompida pelo usuário{Colors.END}")
        return 1
    except Exception as e:
        print(f"\n{Colors.RED}💥 Erro inesperado na validação: {str(e)}{Colors.END}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)