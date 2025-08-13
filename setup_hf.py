#!/usr/bin/env python3
"""
Setup script para Hugging Face Spaces
Este script prepara o ambiente para o sistema Lotomania AI
"""

import os
import sys
import subprocess
import asyncio
from pathlib import Path

def install_requirements():
    """Instala dependências adicionais se necessário"""
    try:
        # Verificar se torch está instalado corretamente
        import torch
        print("✅ PyTorch OK")
    except ImportError:
        print("📦 Instalando PyTorch CPU...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cpu"])

def create_directories():
    """Cria diretórios necessários"""
    directories = [
        "data",
        "models", 
        "logs",
        "cache",
        ".streamlit"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"📁 Diretório criado: {dir_name}")

def setup_environment():
    """Configura variáveis de ambiente"""
    # Configurações para HF Spaces
    os.environ.setdefault("PYTHONPATH", ".")
    os.environ.setdefault("STREAMLIT_SERVER_PORT", "7860")
    os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
    
    print("⚙️ Variáveis de ambiente configuradas")

def check_secrets():
    """Verifica se secrets estão configurados"""
    try:
        import streamlit as st
        
        required_secrets = ["MONGODB_URI"]
        missing_secrets = []
        
        for secret in required_secrets:
            if secret not in st.secrets:
                missing_secrets.append(secret)
        
        if missing_secrets:
            print(f"⚠️ Secrets faltando: {missing_secrets}")
            print("Configure em Settings > Repository secrets no HF Spaces")
            return False
        else:
            print("✅ Todos os secrets configurados")
            return True
            
    except Exception as e:
        print(f"ℹ️ Não foi possível verificar secrets: {e}")
        return True  # Continuar mesmo assim

def main():
    """Função principal de setup"""
    print("🚀 Configurando Lotomania AI para Hugging Face Spaces...")
    
    # 1. Instalar dependências
    install_requirements()
    
    # 2. Criar diretórios
    create_directories()
    
    # 3. Configurar ambiente
    setup_environment()
    
    # 4. Verificar secrets
    secrets_ok = check_secrets()
    
    print("\n" + "="*50)
    print("✅ Setup completo!")
    print("🎯 Sistema Lotomania AI pronto para uso")
    
    if not secrets_ok:
        print("⚠️ Configure os secrets necessários antes de usar")
    
    print("="*50)

if __name__ == "__main__":
    main()