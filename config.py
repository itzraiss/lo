import os
from typing import Dict, List
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DatabaseConfig:
    """Configurações do banco de dados"""
    uri: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
    db_name: str = os.getenv("MONGODB_DB", "lotomania")
    
    # Collections
    contests_collection: str = "contests"
    predictions_collection: str = "predictions"
    models_collection: str = "model_runs"
    logs_collection: str = "logs"
    metrics_collection: str = "metrics"

@dataclass
class APIConfig:
    """Configurações da API da Caixa"""
    caixa_url: str = os.getenv("CAIXA_API_URL", 
                              "https://servicebus2.caixa.gov.br/portaldeloterias/api/home/ultimos-resultados")
    update_interval_hours: int = int(os.getenv("UPDATE_INTERVAL_HOURS", "6"))
    timeout_seconds: int = 30
    max_retries: int = 3

@dataclass
class ModelConfig:
    """Configurações dos modelos de ML"""
    # Thresholds de confiança
    confidence_threshold_18: float = float(os.getenv("CONFIDENCE_THRESHOLD_18", "0.02"))
    confidence_threshold_17: float = float(os.getenv("CONFIDENCE_THRESHOLD_17", "0.10"))
    confidence_threshold_16: float = float(os.getenv("CONFIDENCE_THRESHOLD_16", "0.50"))
    
    # Parâmetros de geração de jogos
    numbers_per_ticket: int = 50
    total_tickets: int = 3
    max_overlap: int = int(os.getenv("MAX_OVERLAP_BETWEEN_TICKETS", "40"))
    candidate_pool_size: int = int(os.getenv("CANDIDATE_POOL_SIZE", "80"))
    
    # Monte Carlo
    monte_carlo_sims: int = int(os.getenv("MONTE_CARLO_SIMULATIONS", "100000"))
    
    # Feature engineering
    lookback_window: int = 100
    min_training_contests: int = 500
    
    # Model ensemble weights
    model_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.model_weights is None:
            self.model_weights = {
                "statistical": 0.25,
                "lightgbm": 0.30,
                "xgboost": 0.25,
                "lstm": 0.20
            }

@dataclass
class PerformanceConfig:
    """Configurações de performance"""
    max_workers: int = int(os.getenv("MAX_WORKERS", "2"))
    cache_ttl_hours: int = int(os.getenv("CACHE_TTL_HOURS", "1"))
    enable_gpu: bool = os.getenv("ENABLE_GPU", "false").lower() == "true"
    batch_size: int = 1000
    memory_limit_mb: int = 1024  # Limite para HF Spaces

@dataclass
class LoggingConfig:
    """Configurações de logging"""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    log_to_file: bool = os.getenv("LOG_TO_FILE", "true").lower() == "true"
    log_file: str = "logs/lotomania.log"
    max_file_size: str = "10MB"
    backup_count: int = 5

@dataclass
class SecurityConfig:
    """Configurações de segurança"""
    secret_key: str = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    admin_password: str = os.getenv("ADMIN_PASSWORD", "admin123")
    
@dataclass
class HuggingFaceConfig:
    """Configurações específicas do Hugging Face Spaces"""
    space_id: str = os.getenv("HF_SPACE_ID", "")
    token: str = os.getenv("HF_TOKEN", "")
    auto_sleep_disabled: bool = True
    persistent_storage: bool = False

class Config:
    """Configuração principal do sistema"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.api = APIConfig()
        self.model = ModelConfig()
        self.performance = PerformanceConfig()
        self.logging = LoggingConfig()
        self.security = SecurityConfig()
        self.huggingface = HuggingFaceConfig()
        
        # Lotomania specific
        self.lotomania_numbers = list(range(1, 101))  # 1 to 100
        self.winning_numbers_count = 20
        
    def validate(self) -> List[str]:
        """Valida configurações e retorna lista de erros"""
        errors = []
        
        if not self.database.uri:
            errors.append("MONGODB_URI não configurado")
            
        if self.model.confidence_threshold_18 > 1.0:
            errors.append("Threshold de confiança para 18 acertos inválido")
            
        if self.model.max_overlap >= self.model.numbers_per_ticket:
            errors.append("Overlap máximo deve ser menor que números por cartela")
            
        return errors

# Instância global
config = Config()