"""
Componente de infraestrutura para publicação de dashboards e relatórios.

Este módulo é responsável por gerar e publicar dashboards interativos e
relatórios estáticos para visualização de analytics.
"""
import logging
import os
import json
import asyncio
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
import tempfile
import shutil
import base64
import re
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Environment, FileSystemLoader, select_autoescape
import pdfkit
import pandas as pd


class DashboardPublisher:
    """
    Componente para publicação de dashboards e relatórios.
    
    Esta classe é responsável por gerar relatórios em diferentes formatos
    (PDF, HTML, JSON) e publicá-los para acesso dos usuários.
    """
    
    def __init__(
        self,
        output_dir: str,
        template_dir: str,
        base_url: str,
        logger: Optional[logging.Logger] = None
    ):
        """
        Inicializa o publicador de dashboards.
        
        Args:
            output_dir: Diretório de saída para os relatórios
            template_dir: Diretório com templates para relatórios
            base_url: URL base para acesso aos relatórios
            logger: Logger opcional
        """
        self.output_dir = output_dir
        self.template_dir = template_dir
        self.base_url = base_url
        self.logger = logger or logging.getLogger(__name__)
        
        # Inicializar ambiente Jinja2 para templates
        self.jinja_env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=select_autoescape(['html', 'xml'])
        )
        
        # Garantir que o diretório de saída existe
        os.makedirs(output_dir, exist_ok=True)
    
    async def publish_report(
        self,
        user_id: str,
        report_id: str,
        report_name: str,
        report_data: Dict[str, Any],
        format: str = "pdf",
        filename: Optional[str] = None
    ) -> str:
        """
        Publica um relatório.
        
        Args:
            user_id: ID do usuário
            report_id: ID do relatório
            report_name: Nome do relatório
            report_data: Dados para o relatório
            format: Formato do relatório (pdf, html, json)
            filename: Nome do arquivo (opcional)
            
        Returns:
            URL do relatório publicado
        """
        self.logger.info(f"Publicando relatório '{report_name}' para usuário {user_id}")
        
        # Criar diretório para o usuário se não existir
        user_dir = os.path.join(self.output_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Determinar nome do arquivo
        if not filename:
            filename = f"{report_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"
        
        # Publicar no formato solicitado
        if format.lower() == "pdf":
            report_url = await self._publish_pdf_report(
                user_id=user_id,
                report_id=report_id,
                report_name=report_name,
                report_data=report_data,
                filename=filename
            )
        elif format.lower() == "html":
            report_url = await self._publish_html_report(
                user_id=user_id,
                report_id=report_id,
                report_name=report_name,
                report_data=report_data,
                filename=filename
            )
        elif format.lower() == "json":
            report_url = await self._publish_json_report(
                user_id=user_id,
                report_id=report_id,
                report_name=report_name,
                report_data=report_data,
                filename=filename
            )
        else:
            # Formato padrão: PDF
            report_url = await self._publish_pdf_report(
                user_id=user_id,
                report_id=report_id,
                report_name=report_name,
                report_data=report_data,
                filename=filename
            )
        
        return report_url
    
    async def _publish_pdf_report(
        self,
        user_id: str,
        report_id: str,
        report_name: str,
        report_data: Dict[str, Any],
        filename: str
    ) -> str:
        """
        Publica um relatório em formato PDF.
        
        Args:
            user_id: ID do usuário
            report_id: ID do relatório
            report_name: Nome do relatório
            report_data: Dados para o relatório
            filename: Nome do arquivo
            
        Returns:
            URL do relatório PDF
        """
        # Determinar tipo de relatório
        report_type = self._determine_report_type(report_data)
        
        # Gerar HTML primeiro
        html_content = await self._generate_report_html(
            report_name=report_name,
            report_type=report_type,
            report_data=report_data
        )
        
        # Caminho para o arquivo PDF
        pdf_path = os.path.join(self.output_dir, user_id, f"{filename}.pdf")
        
        # Gerar PDF a partir do HTML
        try:
            # Criar diretório temporário para arquivos auxiliares (imagens, etc)
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extrair e salvar imagens base64 como arquivos
                html_content = self._extract_and_save_images(html_content, temp_dir)
                
                # Opções para pdfkit
                options = {
                    'page-size': 'A4',
                    'margin-top': '0.75in',
                    'margin-right': '0.75in',
                    'margin-bottom': '0.75in',
                    'margin-left': '0.75in',
                    'encoding': 'UTF-8',
                    'no-outline': None,
                    'enable-local-file-access': None
                }
                
                # Gerar PDF
                pdfkit.from_string(html_content, pdf_path, options=options)
                
                self.logger.info(f"Relatório PDF gerado com sucesso: {pdf_path}")
        except Exception as e:
            self.logger.error(f"Erro ao gerar PDF: {e}")
            # Criar um PDF simples em caso de erro
            self._generate_simple_pdf(pdf_path, report_name, "Erro ao gerar relatório completo.")
        
        # URL para o arquivo PDF
        return f"{self.base_url}/{user_id}/{filename}.pdf"
    
    async def _publish_html_report(
        self,
        user_id: str,
        report_id: str,
        report_name: str,
        report_data: Dict[str, Any],
        filename: str
    ) -> str:
        """
        Publica um relatório em formato HTML.
        
        Args:
            user_id: ID do usuário
            report_id: ID do relatório
            report_name: Nome do relatório
            report_data: Dados para o relatório
            filename: Nome do arquivo
            
        Returns:
            URL do relatório HTML
        """
        # Determinar tipo de relatório
        report_type = self._determine_report_type(report_data)
        
        # Gerar HTML
        html_content = await self._generate_report_html(
            report_name=report_name,
            report_type=report_type,
            report_data=report_data
        )
        
        # Caminho para o arquivo HTML
        html_path = os.path.join(self.output_dir, user_id, f"{filename}.html")
        
        # Salvar HTML
        try:
            # Criar diretório para imagens
            images_dir = os.path.join(self.output_dir, user_id, f"{filename}_files")
            os.makedirs(images_dir, exist_ok=True)
            
            # Extrair e salvar imagens base64 como arquivos
            html_content = self._extract_and_save_images(
                html_content, 
                images_dir, 
                url_prefix=f"{filename}_files/"
            )
            
            # Salvar arquivo HTML
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            self.logger.info(f"Relatório HTML gerado com sucesso: {html_path}")
        except Exception as e:
            self.logger.error(f"Erro ao gerar HTML: {e}")
            # Criar um HTML simples em caso de erro
            self._generate_simple_html(html_path, report_name, "Erro ao gerar relatório completo.")
        
        # URL para o arquivo HTML
        return f"{self.base_url}/{user_id}/{filename}.html"
    
    async def _publish_json_report(
        self,
        user_id: str,
        report_id: str,
        report_name: str,
        report_data: Dict[str, Any],
        filename: str
    ) -> str:
        """
        Publica um relatório em formato JSON.
        
        Args:
            user_id: ID do usuário
            report_id: ID do relatório
            report_name: Nome do relatório
            report_data: Dados para o relatório
            filename: Nome do arquivo
            
        Returns:
            URL do relatório JSON
        """
        # Caminho para o arquivo JSON
        json_path = os.path.join(self.output_dir, user_id, f"{filename}.json")
        
        # Preparar dados para JSON
        json_data = {
            "id": report_id,
            "name": report_name,
            "created_at": datetime.utcnow().isoformat(),
            "data": report_data
        }
        
        # Função para serializar tipos não-JSON
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, (Decimal, float)):
                return float(obj)
            raise TypeError(f"Type {type(obj)} not serializable")
        
        # Salvar JSON
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, default=json_serializer, indent=2)
                
            self.logger.info(f"Relatório JSON gerado com sucesso: {json_path}")
        except Exception as e:
            self.logger.error(f"Erro ao gerar JSON: {e}")
            # Criar um JSON simples em caso de erro
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "id": report_id,
                    "name": report_name,
                    "error": str(e)
                }, f)
        
        # URL para o arquivo JSON
        return f"{self.base_url}/{user_id}/{filename}.json"
    
    async def _generate_report_html(
        self,
        report_name: str,
        report_type: str,
        report_data: Dict[str, Any]
    ) -> str:
        """
        Gera o conteúdo HTML para um relatório.
        
        Args:
            report_name: Nome do relatório
            report_type: Tipo do relatório
            report_data: Dados para o relatório
            
        Returns:
            Conteúdo HTML do relatório
        """
        # Selecionar template com base no tipo de relatório
        template_name = f"{report_type}_report.html"
        
        try:
            template = self.jinja_env.get_template(template_name)
        except Exception as e:
            self.logger.error(f"Erro ao carregar template {template_name}: {e}")
            # Usar template padrão em caso de erro
            template = self.jinja_env.get_template("default_report.html")
        
        # Renderizar template com os dados
        try:
            html_content = template.render(
                report_name=report_name,
                report_type=report_type,
                current_date=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
                data=report_data
            )
            
            return html_content
        except Exception as e:
            self.logger.error(f"Erro ao renderizar template: {e}")
            # Retornar HTML básico em caso de erro
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>{report_name}</title>
            </head>
            <body>
                <h1>{report_name}</h1>
                <p>Erro ao gerar relatório: {str(e)}</p>
            </body>
            </html>
            """
    
    def _determine_report_type(self, report_data: Dict[str, Any]) -> str:
        """
        Determina o tipo de relatório com base nos dados.
        
        Args:
            report_data: Dados do relatório
            
        Returns:
            Tipo do relatório
        """
        # Verificar campos específicos para identificar o tipo
        if "performance_analysis" in report_data:
            return "performance_summary"
        elif "performance_analyses" in report_data:
            return "strategy_comparison"
        elif "market_data" in report_data:
            return "market_analysis"
        elif "trades_data" in report_data:
            return "trade_analysis"
        elif "portfolio_summary" in report_data:
            return "portfolio_overview"
        else:
            return "default"
    
    def _extract_and_save_images(
        self,
        html_content: str,
        output_dir: str,
        url_prefix: str = ""
    ) -> str:
        """
        Extrai imagens base64 do HTML e as salva como arquivos.
        
        Args:
            html_content: Conteúdo HTML
            output_dir: Diretório onde salvar as imagens
            url_prefix: Prefixo para URLs das imagens
            
        Returns:
            HTML atualizado com referências aos arquivos de imagem
        """
        # Regex para encontrar imagens base64
        pattern = r'src="data:image/([^;]+);base64,([^"]+)"'
        
        # Função para substituir cada ocorrência
        def replace_image(match):
            img_format = match.group(1)
            img_data = match.group(2)
            
            # Gerar nome de arquivo único
            img_filename = f"image_{uuid.uuid4()}.{img_format}"
            img_path = os.path.join(output_dir, img_filename)
            
            # Salvar imagem
            try:
                with open(img_path, 'wb') as f:
                    f.write(base64.b64decode(img_data))
                
                # Retornar referência à imagem salva
                return f'src="{url_prefix}{img_filename}"'
            except Exception as e:
                self.logger.error(f"Erro ao salvar imagem: {e}")
                # Manter a imagem base64 original em caso de erro
                return match.group(0)
        
        # Substituir todas as ocorrências
        updated_html = re.sub(pattern, replace_image, html_content)
        
        return updated_html
    
    def _generate_simple_pdf(self, path: str, title: str, message: str) -> None:
        """
        Gera um PDF simples com uma mensagem.
        
        Args:
            path: Caminho onde salvar o PDF
            title: Título do PDF
            message: Mensagem a incluir
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
        </head>
        <body>
            <h1>{title}</h1>
            <p>{message}</p>
            <p>Gerado em: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </body>
        </html>
        """
        
        try:
            pdfkit.from_string(html_content, path)
        except Exception as e:
            self.logger.error(f"Erro ao gerar PDF simples: {e}")
    
    def _generate_simple_html(self, path: str, title: str, message: str) -> None:
        """
        Gera um HTML simples com uma mensagem.
        
        Args:
            path: Caminho onde salvar o HTML
            title: Título do HTML
            message: Mensagem a incluir
        """
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
        </head>
        <body>
            <h1>{title}</h1>
            <p>{message}</p>
            <p>Gerado em: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </body>
        </html>
        """
        
        try:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        except Exception as e:
            self.logger.error(f"Erro ao gerar HTML simples: {e}")
    
    async def create_dashboard(
        self,
        user_id: str,
        dashboard_id: str,
        dashboard_name: str,
        layout: Dict[str, Any]
    ) -> str:
        """
        Cria um dashboard interativo.
        
        Args:
            user_id: ID do usuário
            dashboard_id: ID do dashboard
            dashboard_name: Nome do dashboard
            layout: Layout do dashboard (widgets, posições, etc)
            
        Returns:
            URL do dashboard
        """
        self.logger.info(f"Criando dashboard '{dashboard_name}' para usuário {user_id}")
        
        # Criar diretório para o usuário se não existir
        user_dir = os.path.join(self.output_dir, user_id)
        os.makedirs(user_dir, exist_ok=True)
        
        # Diretório para o dashboard
        dashboard_dir = os.path.join(user_dir, f"dashboard_{dashboard_id}")
        os.makedirs(dashboard_dir, exist_ok=True)
        
        # Caminho para o arquivo principal do dashboard
        dashboard_path = os.path.join(dashboard_dir, "index.html")
        
        # Preparar dados para o template
        dashboard_data = {
            "id": dashboard_id,
            "name": dashboard_name,
            "layout": layout,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Renderizar template de dashboard
        try:
            template = self.jinja_env.get_template("dashboard.html")
            html_content = template.render(dashboard=dashboard_data)
            
            # Salvar arquivo HTML
            with open(dashboard_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
            # Copiar arquivos estáticos (JS, CSS)
            static_dir = os.path.join(self.template_dir, "static")
            if os.path.exists(static_dir):
                target_static_dir = os.path.join(dashboard_dir, "static")
                os.makedirs(target_static_dir, exist_ok=True)
                
                # Copiar arquivos
                for filename in os.listdir(static_dir):
                    src_file = os.path.join(static_dir, filename)
                    dst_file = os.path.join(target_static_dir, filename)
                    
                    if os.path.isfile(src_file):
                        shutil.copy2(src_file, dst_file)
            
            self.logger.info(f"Dashboard criado com sucesso: {dashboard_path}")
        except Exception as e:
            self.logger.error(f"Erro ao criar dashboard: {e}")
            # Criar um HTML simples em caso de erro
            self._generate_simple_html(dashboard_path, dashboard_name, f"Erro ao gerar dashboard: {e}")
        
        # URL para o dashboard
        return f"{self.base_url}/{user_id}/dashboard_{dashboard_id}/"
    
    async def update_dashboard_data(
        self,
        user_id: str,
        dashboard_id: str,
        widget_id: str,
        data: Dict[str, Any]
    ) -> bool:
        """
        Atualiza dados de um widget específico em um dashboard.
        
        Args:
            user_id: ID do usuário
            dashboard_id: ID do dashboard
            widget_id: ID do widget
            data: Novos dados para o widget
            
        Returns:
            True se atualizado com sucesso, False caso contrário
        """
        self.logger.info(f"Atualizando dados do widget {widget_id} no dashboard {dashboard_id}")
        
        # Caminho para o diretório de dados do dashboard
        data_dir = os.path.join(self.output_dir, user_id, f"dashboard_{dashboard_id}", "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Caminho para o arquivo de dados do widget
        data_path = os.path.join(data_dir, f"{widget_id}.json")
        
        # Função para serializar tipos não-JSON
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, (Decimal, float)):
                return float(obj)
            raise TypeError(f"Type {type(obj)} not serializable")
        
        # Salvar dados
        try:
            with open(data_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "widget_id": widget_id,
                    "updated_at": datetime.utcnow().isoformat(),
                    "data": data
                }, f, default=json_serializer, indent=2)
                
            self.logger.info(f"Dados do widget atualizados com sucesso: {data_path}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao atualizar dados do widget: {e}")
            return False
    
    async def delete_dashboard(
        self,
        user_id: str,
        dashboard_id: str
    ) -> bool:
        """
        Exclui um dashboard.
        
        Args:
            user_id: ID do usuário
            dashboard_id: ID do dashboard
            
        Returns:
            True se excluído com sucesso, False caso contrário
        """
        self.logger.info(f"Excluindo dashboard {dashboard_id} do usuário {user_id}")
        
        # Caminho para o diretório do dashboard
        dashboard_dir = os.path.join(self.output_dir, user_id, f"dashboard_{dashboard_id}")
        
        # Verificar se existe
        if not os.path.exists(dashboard_dir):
            self.logger.warning(f"Dashboard {dashboard_id} não encontrado para exclusão")
            return False
        
        # Excluir diretório
        try:
            shutil.rmtree(dashboard_dir)
            self.logger.info(f"Dashboard excluído com sucesso: {dashboard_dir}")
            return True
        except Exception as e:
            self.logger.error(f"Erro ao excluir dashboard: {e}")
            return False