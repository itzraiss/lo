import os
import sys
import json
import time
import psutil
from datetime import datetime
from typing import Dict, Any, Optional
from loguru import logger
from config import config
import functools

class PerformanceMonitor:
    """Monitor de performance para otimização contínua"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {}
        
    def log_system_metrics(self):
        """Log métricas do sistema"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_mb": memory.available / (1024 * 1024),
                "disk_percent": disk.percent,
                "uptime_hours": (time.time() - self.start_time) / 3600
            }
            
            logger.info("System metrics", extra={"metrics": metrics})
            return metrics
            
        except Exception as e:
            logger.error(f"Erro ao coletar métricas do sistema: {e}")
            return {}

class LotomaniaLogger:
    """Sistema de logging especializado para Lotomania"""
    
    def __init__(self):
        self.performance_monitor = PerformanceMonitor()
        self._setup_logger()
        
    def _setup_logger(self):
        """Configura o sistema de logging"""
        # Remove handlers padrão
        logger.remove()
        
        # Formato customizado
        log_format = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )
        
        # Console output
        logger.add(
            sys.stderr,
            format=log_format,
            level=config.logging.level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )
        
        # File output se habilitado
        if config.logging.log_to_file:
            os.makedirs(os.path.dirname(config.logging.log_file), exist_ok=True)
            logger.add(
                config.logging.log_file,
                format=log_format,
                level=config.logging.level,
                rotation=config.logging.max_file_size,
                retention=config.logging.backup_count,
                compression="gz",
                serialize=True  # JSON format for structured logging
            )
    
    def log_prediction_cycle(self, cycle_data: Dict[str, Any]):
        """Log de um ciclo completo de predição"""
        logger.info(
            "Ciclo de predição completado",
            extra={
                "cycle_data": cycle_data,
                "system_metrics": self.performance_monitor.log_system_metrics()
            }
        )
    
    def log_model_training(self, model_name: str, metrics: Dict[str, float], duration: float):
        """Log de treinamento de modelo"""
        logger.info(
            f"Modelo {model_name} treinado",
            extra={
                "model_name": model_name,
                "training_metrics": metrics,
                "duration_seconds": duration,
                "memory_usage": psutil.virtual_memory().percent
            }
        )
    
    def log_api_call(self, endpoint: str, status_code: int, duration: float, data_size: int = 0):
        """Log de chamadas de API"""
        logger.info(
            f"API call: {endpoint}",
            extra={
                "endpoint": endpoint,
                "status_code": status_code,
                "duration_ms": duration * 1000,
                "data_size_bytes": data_size
            }
        )
    
    def log_confidence_check(self, probabilities: Dict[str, float], passed: bool, tickets_generated: bool):
        """Log de verificação de confiança"""
        logger.info(
            "Verificação de confiança",
            extra={
                "probabilities": probabilities,
                "confidence_passed": passed,
                "tickets_generated": tickets_generated,
                "thresholds": {
                    "18_hits": config.model.confidence_threshold_18,
                    "17_hits": config.model.confidence_threshold_17,
                    "16_hits": config.model.confidence_threshold_16
                }
            }
        )
    
    def log_monte_carlo_simulation(self, results: Dict[str, Any], duration: float):
        """Log de simulação Monte Carlo"""
        logger.info(
            "Simulação Monte Carlo completada",
            extra={
                "simulation_results": results,
                "duration_seconds": duration,
                "simulations_count": config.model.monte_carlo_sims
            }
        )
    
    def log_optimization_result(self, method: str, tickets: list, objective_value: float, duration: float):
        """Log de resultado de otimização"""
        logger.info(
            f"Otimização {method} completada",
            extra={
                "optimization_method": method,
                "tickets_count": len(tickets),
                "objective_value": objective_value,
                "duration_seconds": duration,
                "overlap_analysis": self._analyze_ticket_overlap(tickets)
            }
        )
    
    def _analyze_ticket_overlap(self, tickets: list) -> Dict[str, int]:
        """Analisa overlap entre tickets"""
        if len(tickets) < 2:
            return {}
        
        overlaps = {}
        for i in range(len(tickets)):
            for j in range(i + 1, len(tickets)):
                key = f"ticket_{i+1}_vs_{j+1}"
                overlap = len(set(tickets[i]) & set(tickets[j]))
                overlaps[key] = overlap
        
        return overlaps
    
    def log_error(self, error: Exception, context: str, extra_data: Optional[Dict] = None):
        """Log de erro com contexto"""
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context
        }
        
        if extra_data:
            error_data.update(extra_data)
        
        logger.error(
            f"Erro em {context}",
            extra={"error_data": error_data}
        )

def performance_logger(func):
    """Decorator para logging automático de performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent
        
        try:
            result = func(*args, **kwargs)
            success = True
            error_msg = None
        except Exception as e:
            result = None
            success = False
            error_msg = str(e)
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.virtual_memory().percent
            duration = end_time - start_time
            
            logger.info(
                f"Function {func.__name__} executed",
                extra={
                    "function_name": func.__name__,
                    "duration_seconds": duration,
                    "memory_start_percent": start_memory,
                    "memory_end_percent": end_memory,
                    "memory_delta_percent": end_memory - start_memory,
                    "success": success,
                    "error_message": error_msg
                }
            )
        
        return result
    return wrapper

def async_performance_logger(func):
    """Decorator para logging de performance de funções async"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = psutil.virtual_memory().percent
        
        try:
            result = await func(*args, **kwargs)
            success = True
            error_msg = None
        except Exception as e:
            result = None
            success = False
            error_msg = str(e)
            raise
        finally:
            end_time = time.time()
            end_memory = psutil.virtual_memory().percent
            duration = end_time - start_time
            
            logger.info(
                f"Async function {func.__name__} executed",
                extra={
                    "function_name": func.__name__,
                    "duration_seconds": duration,
                    "memory_start_percent": start_memory,
                    "memory_end_percent": end_memory,
                    "memory_delta_percent": end_memory - start_memory,
                    "success": success,
                    "error_message": error_msg
                }
            )
        
        return result
    return wrapper

# Instância global
loto_logger = LotomaniaLogger()

# Convenience functions
def log_info(message: str, **kwargs):
    logger.info(message, extra=kwargs)

def log_warning(message: str, **kwargs):
    logger.warning(message, extra=kwargs)

def log_error(message: str, **kwargs):
    logger.error(message, extra=kwargs)

def log_debug(message: str, **kwargs):
    logger.debug(message, extra=kwargs)