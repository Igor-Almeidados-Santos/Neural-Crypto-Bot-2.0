"""
Script de validação para verificar se as correções da Fase 1 e 1.5 foram aplicadas.
Execução: poetry run python scripts/validate_phase1_fixes.py
"""
import sys
from pathlib import Path
import subprocess

# Define o diretório raiz do projeto
ROOT_DIR = Path(__file__).parent.parent
SRC_DIR = ROOT_DIR / "src"

class Validator:
    def __init__(self):
        self.errors = 0
        self.success = 0

    def check(self, description: str, condition: bool):
        if condition:
            print(f"✅ [SUCESSO] {description}")
            self.success += 1
        else:
            print(f"❌ [FALHA]   {description}")
            self.errors += 1

    def run(self):
        print("--- Iniciando Validação da Refatoração (Fase 1 e 1.5) ---")

        # 1. Verificar se o Makefile existe
        self.check("Makefile existe na raiz do projeto.", (ROOT_DIR / "Makefile").is_file())

        # 2. Verificar se o pyproject.toml é de Poetry
        pyproject_content = (ROOT_DIR / "pyproject.toml").read_text()
        self.check("pyproject.toml contém a seção [tool.poetry].", "[tool.poetry]" in pyproject_content)

        # 3. Verificar se o 'src' foi removido dos imports
        # Checa um arquivo crítico como exemplo
        api_main_content = (SRC_DIR / "api" / "main.py").read_text()
        self.check("Import em 'api/main.py' não contém 'from src.'.", "from src." not in api_main_content)
        
        # 4. Verificar se a configuração centralizada está sendo usada
        self.check("'api/main.py' importa o objeto 'settings' centralizado.", "from common.utils.settings import settings" in api_main_content)
        
        # 5. Verificar se o antigo config.py foi removido
        self.check("O antigo 'common/utils/config.py' foi removido.", not (SRC_DIR / "common" / "utils" / "config.py").exists())

        # 6. Verificar se os scripts antigos foram removidos
        self.check("O script 'scripts/docs' foi removido.", not (ROOT_DIR / "scripts" / "docs").exists())
        self.check("O script 'setup.sh' foi removido da raiz.", not (ROOT_DIR / "setup.sh").exists())

        # 7. Verificar se o poetry está funcionando
        try:
            result = subprocess.run(["poetry", "check"], capture_output=True, text=True, check=True)
            self.check("Comando 'poetry check' é executado com sucesso.", "All set!" in result.stdout)
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Erro ao executar 'poetry check': {e}")
            self.check("Comando 'poetry check' é executado com sucesso.", False)


        print("--- Validação Concluída ---")
        print(f"Resultados: {self.success} sucessos, {self.errors} falhas.")
        
        if self.errors > 0:
            sys.exit(1)
        else:
            print("🎉 Excelente! A fundação do projeto está sólida e bem estruturada.")
            sys.exit(0)

if __name__ == "__main__":
    Validator().run()