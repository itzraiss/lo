import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import asdict
import motor.motor_asyncio
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError
from diskcache import Cache
import hashlib

from config import config
from utils.logger import loto_logger, performance_logger, async_performance_logger

class MongoDBClient:
    """Cliente MongoDB otimizado para Lotomania com cache e pooling"""
    
    def __init__(self):
        self.client = None
        self.async_client = None
        self.db = None
        self.async_db = None
        self.cache = Cache('/tmp/lotomania_cache', size_limit=100 * 1024 * 1024)  # 100MB cache
        self._connected = False
        
    @async_performance_logger
    async def connect(self):
        """Conecta ao MongoDB com configuração otimizada"""
        try:
            # Cliente síncrono
            self.client = MongoClient(
                config.database.uri,
                maxPoolSize=config.performance.max_workers * 2,
                minPoolSize=1,
                maxIdleTimeMS=30000,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000,
                retryWrites=True,
                retryReads=True
            )
            
            # Cliente assíncrono para operações concorrentes
            self.async_client = motor.motor_asyncio.AsyncIOMotorClient(
                config.database.uri,
                maxPoolSize=config.performance.max_workers * 2,
                minPoolSize=1,
                maxIdleTimeMS=30000,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
                socketTimeoutMS=5000
            )
            
            # Databases
            self.db = self.client[config.database.db_name]
            self.async_db = self.async_client[config.database.db_name]
            
            # Teste de conexão
            await self.async_client.admin.command('ping')
            self._connected = True
            
            # Criar indexes
            await self._create_indexes()
            
            loto_logger.log_info("MongoDB conectado com sucesso")
            
        except Exception as e:
            loto_logger.log_error(e, "Erro ao conectar MongoDB")
            raise
    
    async def _create_indexes(self):
        """Cria indexes otimizados para performance"""
        try:
            # Indexes para contests
            await self.async_db[config.database.contests_collection].create_index([
                ("contest", ASCENDING)
            ], unique=True)
            
            await self.async_db[config.database.contests_collection].create_index([
                ("date", DESCENDING)
            ])
            
            # Indexes para predictions
            await self.async_db[config.database.predictions_collection].create_index([
                ("generated_at", DESCENDING)
            ])
            
            await self.async_db[config.database.predictions_collection].create_index([
                ("model_version", ASCENDING)
            ])
            
            # Indexes para model_runs
            await self.async_db[config.database.models_collection].create_index([
                ("timestamp", DESCENDING)
            ])
            
            # Indexes para logs
            await self.async_db[config.database.logs_collection].create_index([
                ("timestamp", DESCENDING)
            ])
            
            # Index TTL para logs (manter apenas 30 dias)
            await self.async_db[config.database.logs_collection].create_index(
                "timestamp", 
                expireAfterSeconds=30 * 24 * 3600  # 30 dias
            )
            
        except Exception as e:
            loto_logger.log_error(e, "Erro ao criar indexes")
    
    def _get_cache_key(self, collection: str, query: Dict = None, **kwargs) -> str:
        """Gera chave de cache baseada na query"""
        cache_data = {
            "collection": collection,
            "query": query or {},
            "kwargs": kwargs
        }
        cache_str = json.dumps(cache_data, sort_keys=True, default=str)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    @async_performance_logger
    async def insert_contest(self, contest_data: Dict[str, Any]) -> bool:
        """Insere dados de um concurso"""
        try:
            # Validar dados
            required_fields = ['contest', 'date', 'numbers']
            for field in required_fields:
                if field not in contest_data:
                    raise ValueError(f"Campo obrigatório '{field}' não encontrado")
            
            # Adicionar timestamp
            contest_data['created_at'] = datetime.utcnow()
            
            # Inserir
            await self.async_db[config.database.contests_collection].insert_one(contest_data)
            
            # Limpar cache relacionado
            self._clear_contests_cache()
            
            loto_logger.log_info(f"Concurso {contest_data['contest']} inserido com sucesso")
            return True
            
        except DuplicateKeyError:
            loto_logger.log_warning(f"Concurso {contest_data['contest']} já existe")
            return False
        except Exception as e:
            loto_logger.log_error(e, f"Erro ao inserir concurso {contest_data.get('contest', 'N/A')}")
            return False
    
    @async_performance_logger
    async def get_contests(self, 
                          limit: Optional[int] = None, 
                          start_date: Optional[datetime] = None,
                          end_date: Optional[datetime] = None,
                          use_cache: bool = True) -> List[Dict[str, Any]]:
        """Busca concursos com cache otimizado"""
        
        # Preparar query
        query = {}
        if start_date or end_date:
            date_filter = {}
            if start_date:
                date_filter['$gte'] = start_date
            if end_date:
                date_filter['$lte'] = end_date
            query['date'] = date_filter
        
        # Verificar cache
        cache_key = self._get_cache_key(
            config.database.contests_collection, 
            query, 
            limit=limit
        )
        
        if use_cache:
            cached_result = self.cache.get(cache_key)
            if cached_result:
                return cached_result
        
        try:
            # Buscar no banco
            cursor = self.async_db[config.database.contests_collection].find(query)
            cursor = cursor.sort("date", DESCENDING)
            
            if limit:
                cursor = cursor.limit(limit)
            
            contests = await cursor.to_list(length=None)
            
            # Converter ObjectId para string
            for contest in contests:
                contest['_id'] = str(contest['_id'])
            
            # Cache result
            if use_cache:
                self.cache.set(cache_key, contests, expire=config.performance.cache_ttl_hours * 3600)
            
            return contests
            
        except Exception as e:
            loto_logger.log_error(e, "Erro ao buscar concursos")
            return []
    
    @async_performance_logger
    async def get_latest_contest(self) -> Optional[Dict[str, Any]]:
        """Busca o concurso mais recente"""
        contests = await self.get_contests(limit=1)
        return contests[0] if contests else None
    
    @async_performance_logger
    async def insert_prediction(self, prediction_data: Dict[str, Any]) -> str:
        """Insere dados de predição"""
        try:
            prediction_data['generated_at'] = datetime.utcnow()
            
            result = await self.async_db[config.database.predictions_collection].insert_one(prediction_data)
            
            # Limpar cache
            self._clear_predictions_cache()
            
            loto_logger.log_info(f"Predição inserida: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            loto_logger.log_error(e, "Erro ao inserir predição")
            raise
    
    @async_performance_logger
    async def get_latest_prediction(self) -> Optional[Dict[str, Any]]:
        """Busca a predição mais recente"""
        try:
            prediction = await self.async_db[config.database.predictions_collection].find_one(
                sort=[("generated_at", DESCENDING)]
            )
            
            if prediction:
                prediction['_id'] = str(prediction['_id'])
                
            return prediction
            
        except Exception as e:
            loto_logger.log_error(e, "Erro ao buscar predição mais recente")
            return None
    
    @async_performance_logger
    async def insert_model_run(self, model_data: Dict[str, Any]) -> str:
        """Insere dados de execução de modelo"""
        try:
            model_data['timestamp'] = datetime.utcnow()
            
            result = await self.async_db[config.database.models_collection].insert_one(model_data)
            
            loto_logger.log_info(f"Model run inserido: {result.inserted_id}")
            return str(result.inserted_id)
            
        except Exception as e:
            loto_logger.log_error(e, "Erro ao inserir model run")
            raise
    
    @async_performance_logger
    async def log_to_db(self, log_data: Dict[str, Any]):
        """Insere log no banco"""
        try:
            log_data['timestamp'] = datetime.utcnow()
            
            await self.async_db[config.database.logs_collection].insert_one(log_data)
            
        except Exception as e:
            # Evitar loop infinito de logs
            print(f"Erro ao inserir log no banco: {e}")
    
    @async_performance_logger
    async def insert_metrics(self, metrics_data: Dict[str, Any]) -> str:
        """Insere métricas de performance"""
        try:
            metrics_data['timestamp'] = datetime.utcnow()
            
            result = await self.async_db[config.database.metrics_collection].insert_one(metrics_data)
            
            return str(result.inserted_id)
            
        except Exception as e:
            loto_logger.log_error(e, "Erro ao inserir métricas")
            raise
    
    @async_performance_logger
    async def get_backtest_results(self, 
                                 model_version: Optional[str] = None,
                                 limit: int = 100) -> List[Dict[str, Any]]:
        """Busca resultados de backtest"""
        try:
            query = {"type": "backtest"}
            if model_version:
                query["model_version"] = model_version
            
            cursor = self.async_db[config.database.metrics_collection].find(query)
            cursor = cursor.sort("timestamp", DESCENDING).limit(limit)
            
            results = await cursor.to_list(length=None)
            
            for result in results:
                result['_id'] = str(result['_id'])
                
            return results
            
        except Exception as e:
            loto_logger.log_error(e, "Erro ao buscar resultados de backtest")
            return []
    
    def _clear_contests_cache(self):
        """Limpa cache de concursos"""
        cache_keys = [key for key in self.cache if config.database.contests_collection in key]
        for key in cache_keys:
            del self.cache[key]
    
    def _clear_predictions_cache(self):
        """Limpa cache de predições"""
        cache_keys = [key for key in self.cache if config.database.predictions_collection in key]
        for key in cache_keys:
            del self.cache[key]
    
    @async_performance_logger
    async def cleanup_old_data(self):
        """Limpeza automática de dados antigos"""
        try:
            # Remover predições antigas (manter apenas últimas 1000)
            predictions_count = await self.async_db[config.database.predictions_collection].count_documents({})
            
            if predictions_count > 1000:
                # Buscar IDs das predições mais antigas
                old_predictions = await self.async_db[config.database.predictions_collection].find(
                    {}, {"_id": 1}
                ).sort("generated_at", ASCENDING).limit(predictions_count - 1000).to_list(length=None)
                
                old_ids = [p["_id"] for p in old_predictions]
                
                await self.async_db[config.database.predictions_collection].delete_many(
                    {"_id": {"$in": old_ids}}
                )
                
                loto_logger.log_info(f"Removidas {len(old_ids)} predições antigas")
            
            # Remover model runs antigos (manter apenas últimos 30 dias)
            thirty_days_ago = datetime.utcnow() - timedelta(days=30)
            result = await self.async_db[config.database.models_collection].delete_many(
                {"timestamp": {"$lt": thirty_days_ago}}
            )
            
            if result.deleted_count > 0:
                loto_logger.log_info(f"Removidos {result.deleted_count} model runs antigos")
            
            # Limpar cache
            self.cache.clear()
            
        except Exception as e:
            loto_logger.log_error(e, "Erro na limpeza de dados antigos")
    
    @performance_logger
    def close(self):
        """Fecha conexões"""
        if self.client:
            self.client.close()
        if self.async_client:
            self.async_client.close()
        self._connected = False
        loto_logger.log_info("Conexões MongoDB fechadas")
    
    @property
    def is_connected(self) -> bool:
        """Verifica se está conectado"""
        return self._connected

# Instância global
mongodb_client = MongoDBClient()