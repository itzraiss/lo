import asyncio
import aiohttp
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import time
import re

from config import config
from database.mongodb_client import mongodb_client
from utils.logger import loto_logger, async_performance_logger, performance_logger

class LotomaniaDataLoader:
    """Sistema de ingestão de dados da Lotomania otimizado para performance"""
    
    def __init__(self):
        self.session = None
        self.last_api_call = 0
        self.api_rate_limit = 1  # 1 segundo entre chamadas
        
    async def __aenter__(self):
        """Context manager para sessão HTTP"""
        timeout = aiohttp.ClientTimeout(total=config.api.timeout_seconds)
        self.session = aiohttp.ClientSession(timeout=timeout)
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Fecha sessão HTTP"""
        if self.session:
            await self.session.close()
    
    @performance_logger
    def load_historical_data(self, file_path: str) -> List[Dict[str, Any]]:
        """Carrega dados históricos do arquivo XLSX"""
        try:
            loto_logger.log_info(f"Carregando dados históricos de {file_path}")
            
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Arquivo {file_path} não encontrado")
            
            # Tentar múltiplos formatos de arquivo
            try:
                df = pd.read_excel(file_path)
            except Exception:
                # Tentar CSV
                df = pd.read_csv(file_path)
            
            contests = []
            
            for _, row in df.iterrows():
                try:
                    contest_data = self._parse_contest_row(row)
                    if contest_data:
                        contests.append(contest_data)
                except Exception as e:
                    loto_logger.log_warning(f"Erro ao processar linha {row.name}: {e}")
                    continue
            
            loto_logger.log_info(f"Carregados {len(contests)} concursos do arquivo histórico")
            return contests
            
        except Exception as e:
            loto_logger.log_error(e, f"Erro ao carregar dados históricos de {file_path}")
            return []
    
    def _parse_contest_row(self, row: pd.Series) -> Optional[Dict[str, Any]]:
        """Converte linha do DataFrame em dados de concurso"""
        try:
            # Mapear colunas comuns
            contest_num = None
            date_val = None
            numbers = []
            
            # Detectar número do concurso
            for col in row.index:
                col_lower = str(col).lower()
                if any(word in col_lower for word in ['concurso', 'contest', 'numero']):
                    contest_num = int(row[col])
                    break
            
            # Detectar data
            for col in row.index:
                col_lower = str(col).lower()
                if any(word in col_lower for word in ['data', 'date']):
                    date_val = pd.to_datetime(row[col])
                    break
            
            # Detectar números sorteados
            for col in row.index:
                col_lower = str(col).lower()
                # Procurar por colunas de dezenas ou números
                if any(word in col_lower for word in ['dezena', 'numero', 'ball', 'bola']):
                    try:
                        num = int(row[col])
                        if 1 <= num <= 100:
                            numbers.append(num)
                    except (ValueError, TypeError):
                        continue
            
            # Se não encontrou números em colunas específicas, tentar detectar automaticamente
            if not numbers:
                for col in row.index:
                    try:
                        val = row[col]
                        if pd.isna(val):
                            continue
                        
                        # Tentar converter para int
                        if isinstance(val, (int, float)):
                            num = int(val)
                            if 1 <= num <= 100:
                                numbers.append(num)
                        elif isinstance(val, str):
                            # Tentar extrair números da string
                            found_nums = re.findall(r'\b([1-9]\d?|100)\b', val)
                            for num_str in found_nums:
                                num = int(num_str)
                                if 1 <= num <= 100:
                                    numbers.append(num)
                    except (ValueError, TypeError):
                        continue
            
            # Validar dados
            if not contest_num or not date_val or len(numbers) != 20:
                return None
            
            # Garantir que números estão únicos e ordenados
            numbers = sorted(list(set(numbers)))
            
            if len(numbers) != 20:
                return None
            
            return {
                'contest': contest_num,
                'date': date_val.strftime('%Y-%m-%d'),
                'numbers': numbers,
                'raw': row.to_dict()
            }
            
        except Exception as e:
            loto_logger.log_debug(f"Erro ao processar linha: {e}")
            return None
    
    @async_performance_logger
    async def fetch_latest_results(self, max_retries: int = None) -> List[Dict[str, Any]]:
        """Busca resultados mais recentes da API da Caixa"""
        if max_retries is None:
            max_retries = config.api.max_retries
            
        for attempt in range(max_retries):
            try:
                # Rate limiting
                time_since_last = time.time() - self.last_api_call
                if time_since_last < self.api_rate_limit:
                    await asyncio.sleep(self.api_rate_limit - time_since_last)
                
                start_time = time.time()
                
                async with self.session.get(config.api.caixa_url) as response:
                    self.last_api_call = time.time()
                    duration = self.last_api_call - start_time
                    
                    loto_logger.log_api_call(
                        config.api.caixa_url,
                        response.status,
                        duration,
                        len(await response.read())
                    )
                    
                    if response.status == 200:
                        data = await response.json()
                        contests = self._parse_api_response(data)
                        return contests
                    else:
                        loto_logger.log_warning(f"API retornou status {response.status}")
                        
            except asyncio.TimeoutError:
                loto_logger.log_warning(f"Timeout na tentativa {attempt + 1}")
            except Exception as e:
                loto_logger.log_error(e, f"Erro na API (tentativa {attempt + 1})")
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                await asyncio.sleep(wait_time)
        
        loto_logger.log_error("Falha ao buscar dados da API após todas as tentativas", "fetch_latest_results")
        return []
    
    def _parse_api_response(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Converte resposta da API em dados estruturados"""
        try:
            contests = []
            
            # A API da Caixa pode retornar diferentes estruturas
            if 'lotomania' in data:
                lotomania_data = data['lotomania']
                
                if isinstance(lotomania_data, list):
                    for contest_data in lotomania_data:
                        contest = self._parse_api_contest(contest_data)
                        if contest:
                            contests.append(contest)
                elif isinstance(lotomania_data, dict):
                    contest = self._parse_api_contest(lotomania_data)
                    if contest:
                        contests.append(contest)
            
            # Tentar outras estruturas possíveis
            elif isinstance(data, list):
                for item in data:
                    if 'modalidade' in item and item.get('modalidade') == 'Lotomania':
                        contest = self._parse_api_contest(item)
                        if contest:
                            contests.append(contest)
            
            loto_logger.log_info(f"Processados {len(contests)} concursos da API")
            return contests
            
        except Exception as e:
            loto_logger.log_error(e, "Erro ao processar resposta da API")
            return []
    
    def _parse_api_contest(self, contest_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Converte dados de um concurso da API"""
        try:
            # Extrair número do concurso
            contest_num = None
            for key in ['concurso', 'numero', 'numeroSorteio']:
                if key in contest_data:
                    contest_num = int(contest_data[key])
                    break
            
            # Extrair data
            date_val = None
            for key in ['dataApuracao', 'data', 'dataResultado']:
                if key in contest_data:
                    date_str = contest_data[key]
                    try:
                        # Tentar formatos comuns
                        if '/' in date_str:
                            date_val = datetime.strptime(date_str, '%d/%m/%Y')
                        elif '-' in date_str:
                            date_val = datetime.strptime(date_str, '%Y-%m-%d')
                        break
                    except ValueError:
                        continue
            
            # Extrair números sorteados
            numbers = []
            for key in ['dezenas', 'numerosSorteados', 'listaDezenas']:
                if key in contest_data:
                    nums_data = contest_data[key]
                    if isinstance(nums_data, list):
                        numbers = [int(n) for n in nums_data if str(n).isdigit()]
                    elif isinstance(nums_data, str):
                        # Tentar extrair números da string
                        numbers = [int(n) for n in re.findall(r'\d+', nums_data) if 1 <= int(n) <= 100]
                    break
            
            # Validar
            if not contest_num or not date_val or len(numbers) != 20:
                return None
            
            return {
                'contest': contest_num,
                'date': date_val.strftime('%Y-%m-%d'),
                'numbers': sorted(numbers),
                'raw': contest_data
            }
            
        except Exception as e:
            loto_logger.log_debug(f"Erro ao processar concurso da API: {e}")
            return None
    
    @async_performance_logger
    async def sync_database(self, contests: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Sincroniza concursos com o banco de dados"""
        inserted = 0
        skipped = 0
        
        try:
            # Processar em batches para otimizar performance
            batch_size = config.performance.batch_size
            
            for i in range(0, len(contests), batch_size):
                batch = contests[i:i + batch_size]
                
                # Processar batch em paralelo
                tasks = [mongodb_client.insert_contest(contest) for contest in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for result in results:
                    if isinstance(result, Exception):
                        loto_logger.log_error(result, "Erro ao inserir concurso em batch")
                        skipped += 1
                    elif result:
                        inserted += 1
                    else:
                        skipped += 1
            
            loto_logger.log_info(f"Sincronização: {inserted} inseridos, {skipped} ignorados")
            return inserted, skipped
            
        except Exception as e:
            loto_logger.log_error(e, "Erro na sincronização com banco de dados")
            return 0, len(contests)
    
    @async_performance_logger
    async def incremental_update(self) -> bool:
        """Atualização incremental com dados da API"""
        try:
            loto_logger.log_info("Iniciando atualização incremental")
            
            # Buscar último concurso no banco
            latest_contest = await mongodb_client.get_latest_contest()
            latest_contest_num = latest_contest['contest'] if latest_contest else 0
            
            # Buscar dados da API
            api_contests = await self.fetch_latest_results()
            
            # Filtrar apenas concursos novos
            new_contests = [
                contest for contest in api_contests 
                if contest['contest'] > latest_contest_num
            ]
            
            if not new_contests:
                loto_logger.log_info("Nenhum concurso novo encontrado")
                return True
            
            # Sincronizar novos concursos
            inserted, skipped = await self.sync_database(new_contests)
            
            success = inserted > 0
            loto_logger.log_info(f"Atualização incremental: {inserted} novos concursos")
            
            return success
            
        except Exception as e:
            loto_logger.log_error(e, "Erro na atualização incremental")
            return False
    
    @async_performance_logger
    async def full_historical_load(self, file_path: str) -> bool:
        """Carregamento completo dos dados históricos"""
        try:
            loto_logger.log_info("Iniciando carregamento histórico completo")
            
            # Carregar dados do arquivo
            contests = self.load_historical_data(file_path)
            
            if not contests:
                loto_logger.log_error("Nenhum dado histórico carregado", "full_historical_load")
                return False
            
            # Sincronizar com banco
            inserted, skipped = await self.sync_database(contests)
            
            # Tentar atualização incremental para dados mais recentes
            await self.incremental_update()
            
            loto_logger.log_info(f"Carregamento histórico completo: {inserted} concursos carregados")
            return inserted > 0
            
        except Exception as e:
            loto_logger.log_error(e, "Erro no carregamento histórico")
            return False
    
    @async_performance_logger
    async def validate_data_integrity(self) -> Dict[str, Any]:
        """Valida integridade dos dados no banco"""
        try:
            contests = await mongodb_client.get_contests()
            
            issues = {
                'duplicates': [],
                'invalid_numbers': [],
                'date_gaps': [],
                'invalid_dates': []
            }
            
            contest_numbers = set()
            previous_date = None
            
            for contest in contests:
                # Verificar duplicatas
                if contest['contest'] in contest_numbers:
                    issues['duplicates'].append(contest['contest'])
                contest_numbers.add(contest['contest'])
                
                # Verificar números válidos
                numbers = contest.get('numbers', [])
                if len(numbers) != 20 or not all(1 <= n <= 100 for n in numbers):
                    issues['invalid_numbers'].append(contest['contest'])
                
                # Verificar datas
                try:
                    contest_date = datetime.strptime(contest['date'], '%Y-%m-%d')
                    if previous_date and (contest_date - previous_date).days > 7:
                        issues['date_gaps'].append(contest['contest'])
                    previous_date = contest_date
                except ValueError:
                    issues['invalid_dates'].append(contest['contest'])
            
            total_issues = sum(len(issue_list) for issue_list in issues.values())
            
            loto_logger.log_info(f"Validação de integridade: {total_issues} problemas encontrados")
            return {
                'total_contests': len(contests),
                'total_issues': total_issues,
                'issues': issues
            }
            
        except Exception as e:
            loto_logger.log_error(e, "Erro na validação de integridade")
            return {'total_contests': 0, 'total_issues': 0, 'issues': {}}

# Funções de conveniência
async def load_historical_data_async(file_path: str) -> bool:
    """Carrega dados históricos de forma assíncrona"""
    async with LotomaniaDataLoader() as loader:
        return await loader.full_historical_load(file_path)

async def update_from_api_async() -> bool:
    """Atualiza dados da API de forma assíncrona"""
    async with LotomaniaDataLoader() as loader:
        return await loader.incremental_update()