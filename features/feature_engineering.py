import numpy as np
import pandas as pd
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import joblib
from scipy import stats
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from diskcache import Cache
import hashlib
import json

from config import config
from database.mongodb_client import mongodb_client
from utils.logger import loto_logger, async_performance_logger, performance_logger

@dataclass
class FeatureConfig:
    """Configuração para engenharia de features"""
    lookback_windows: List[int] = None
    trend_windows: List[int] = None
    fourier_components: int = 5
    pair_analysis_top_k: int = 50
    cache_features: bool = True
    parallel_processing: bool = True
    
    def __post_init__(self):
        if self.lookback_windows is None:
            self.lookback_windows = [10, 20, 50, 100]
        if self.trend_windows is None:
            self.trend_windows = [5, 10, 20]

class LotomaniaFeatureEngineer:
    """Sistema avançado de feature engineering para Lotomania"""
    
    def __init__(self, feature_config: FeatureConfig = None):
        self.config = feature_config or FeatureConfig()
        self.cache = Cache('/tmp/features_cache', size_limit=200 * 1024 * 1024)  # 200MB
        self.scaler = StandardScaler()
        self.min_max_scaler = MinMaxScaler()
        self._feature_cache_keys = set()
        
    @async_performance_logger
    async def engineer_features(self, contests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Gera todas as features de forma otimizada"""
        try:
            loto_logger.log_info("Iniciando engenharia de features")
            
            # Converter para DataFrame para processamento eficiente
            df_contests = self._contests_to_dataframe(contests)
            
            if len(df_contests) < config.model.min_training_contests:
                raise ValueError(f"Necessários pelo menos {config.model.min_training_contests} concursos para feature engineering")
            
            # Gerar features em paralelo
            if self.config.parallel_processing:
                features = await self._generate_features_parallel(df_contests)
            else:
                features = await self._generate_features_sequential(df_contests)
            
            # Cache das features
            if self.config.cache_features:
                self._cache_features(features, contests)
            
            loto_logger.log_info(f"Features geradas: {len(features)} tipos")
            return features
            
        except Exception as e:
            loto_logger.log_error(e, "Erro na engenharia de features")
            raise
    
    def _contests_to_dataframe(self, contests: List[Dict[str, Any]]) -> pd.DataFrame:
        """Converte lista de concursos para DataFrame otimizado"""
        try:
            # Preparar dados para DataFrame
            data = []
            for contest in contests:
                contest_data = {
                    'contest': contest['contest'],
                    'date': pd.to_datetime(contest['date']),
                }
                
                # Adicionar colunas de presença de números (0/1)
                for num in range(1, 101):
                    contest_data[f'num_{num}'] = 1 if num in contest['numbers'] else 0
                
                data.append(contest_data)
            
            df = pd.DataFrame(data)
            df = df.sort_values('date').reset_index(drop=True)
            
            return df
            
        except Exception as e:
            loto_logger.log_error(e, "Erro ao converter concursos para DataFrame")
            raise
    
    async def _generate_features_parallel(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gera features usando processamento paralelo"""
        try:
            # Dividir trabalho em chunks para processamento paralelo
            with ThreadPoolExecutor(max_workers=config.performance.max_workers) as executor:
                
                # Submit todas as tarefas
                futures = {
                    'statistical': executor.submit(self._generate_statistical_features, df),
                    'temporal': executor.submit(self._generate_temporal_features, df),
                    'sequence': executor.submit(self._generate_sequence_features, df),
                    'pair_analysis': executor.submit(self._generate_pair_features, df),
                    'frequency': executor.submit(self._generate_frequency_features, df),
                    'trend': executor.submit(self._generate_trend_features, df),
                    'fourier': executor.submit(self._generate_fourier_features, df),
                    'gap_analysis': executor.submit(self._generate_gap_features, df),
                    'pattern': executor.submit(self._generate_pattern_features, df)
                }
                
                # Coletar resultados
                features = {}
                for feature_type, future in futures.items():
                    try:
                        features[feature_type] = future.result(timeout=300)  # 5 min timeout
                    except Exception as e:
                        loto_logger.log_error(e, f"Erro ao gerar features {feature_type}")
                        features[feature_type] = {}
            
            return features
            
        except Exception as e:
            loto_logger.log_error(e, "Erro no processamento paralelo de features")
            return await self._generate_features_sequential(df)
    
    async def _generate_features_sequential(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gera features sequencialmente (fallback)"""
        features = {}
        
        try:
            features['statistical'] = self._generate_statistical_features(df)
            features['temporal'] = self._generate_temporal_features(df)
            features['sequence'] = self._generate_sequence_features(df)
            features['pair_analysis'] = self._generate_pair_features(df)
            features['frequency'] = self._generate_frequency_features(df)
            features['trend'] = self._generate_trend_features(df)
            features['fourier'] = self._generate_fourier_features(df)
            features['gap_analysis'] = self._generate_gap_features(df)
            features['pattern'] = self._generate_pattern_features(df)
            
        except Exception as e:
            loto_logger.log_error(e, "Erro no processamento sequencial de features")
            
        return features
    
    @performance_logger
    def _generate_statistical_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gera features estatísticas básicas"""
        features = {}
        
        try:
            for num in range(1, 101):
                col = f'num_{num}'
                num_data = df[col].values
                
                # Features básicas
                features[f'{col}_frequency'] = num_data.mean()
                features[f'{col}_std'] = num_data.std()
                features[f'{col}_total_count'] = num_data.sum()
                
                # Features de janelas deslizantes
                for window in self.config.lookback_windows:
                    if len(num_data) >= window:
                        recent_data = num_data[-window:]
                        features[f'{col}_freq_last_{window}'] = recent_data.mean()
                        features[f'{col}_count_last_{window}'] = recent_data.sum()
                        features[f'{col}_trend_last_{window}'] = self._calculate_trend(recent_data)
                
                # Features de distribuição
                features[f'{col}_skewness'] = stats.skew(num_data)
                features[f'{col}_kurtosis'] = stats.kurtosis(num_data)
                
                # Features de percentis
                features[f'{col}_percentile_25'] = np.percentile(num_data, 25)
                features[f'{col}_percentile_75'] = np.percentile(num_data, 75)
                
        except Exception as e:
            loto_logger.log_error(e, "Erro ao gerar features estatísticas")
            
        return features
    
    @performance_logger
    def _generate_temporal_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gera features temporais"""
        features = {}
        
        try:
            df['day_of_week'] = df['date'].dt.dayofweek
            df['day_of_month'] = df['date'].dt.day
            df['month'] = df['date'].dt.month
            df['quarter'] = df['date'].dt.quarter
            df['year'] = df['date'].dt.year
            
            # Features temporais por número
            for num in range(1, 101):
                col = f'num_{num}'
                
                # Análise por dia da semana
                for dow in range(7):
                    mask = df['day_of_week'] == dow
                    if mask.any():
                        features[f'{col}_freq_dow_{dow}'] = df.loc[mask, col].mean()
                
                # Análise por mês
                for month in range(1, 13):
                    mask = df['month'] == month
                    if mask.any():
                        features[f'{col}_freq_month_{month}'] = df.loc[mask, col].mean()
                
                # Análise de sazonalidade
                features[f'{col}_seasonal_strength'] = self._calculate_seasonal_strength(df[col].values)
                
        except Exception as e:
            loto_logger.log_error(e, "Erro ao gerar features temporais")
            
        return features
    
    @performance_logger
    def _generate_sequence_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gera features de sequência para modelos LSTM"""
        features = {}
        
        try:
            # Criar sequências de diferentes tamanhos
            for window in [10, 20, 50]:
                if len(df) >= window:
                    # Matriz de sequências (window x 100)
                    sequences = []
                    for i in range(len(df) - window + 1):
                        sequence = []
                        for num in range(1, 101):
                            sequence.append(df[f'num_{num}'].iloc[i:i+window].values)
                        sequences.append(sequence)
                    
                    features[f'sequences_window_{window}'] = np.array(sequences)
                    
                    # Features agregadas das sequências
                    if sequences:
                        seq_array = np.array(sequences)
                        features[f'seq_mean_window_{window}'] = seq_array.mean(axis=(0, 2))
                        features[f'seq_std_window_{window}'] = seq_array.std(axis=(0, 2))
                        features[f'seq_trend_window_{window}'] = self._calculate_sequence_trends(seq_array)
        
        except Exception as e:
            loto_logger.log_error(e, "Erro ao gerar features de sequência")
            
        return features
    
    @performance_logger
    def _generate_pair_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gera features de análise de pares"""
        features = {}
        
        try:
            # Matriz de coocorrência
            cooccurrence_matrix = np.zeros((100, 100))
            
            for _, row in df.iterrows():
                present_numbers = [num for num in range(1, 101) if row[f'num_{num}'] == 1]
                
                # Atualizar matriz de coocorrência
                for i, num1 in enumerate(present_numbers):
                    for j, num2 in enumerate(present_numbers):
                        if i != j:
                            cooccurrence_matrix[num1-1, num2-1] += 1
            
            # Normalizar matriz
            total_contests = len(df)
            cooccurrence_matrix = cooccurrence_matrix / total_contests
            
            # Features de pares mais frequentes
            for num in range(1, 101):
                # Top pares para cada número
                pairs = cooccurrence_matrix[num-1, :]
                top_pairs_idx = np.argsort(pairs)[-self.config.pair_analysis_top_k:]
                
                features[f'num_{num}_top_pair_strength'] = pairs[top_pairs_idx].mean()
                features[f'num_{num}_pair_variance'] = pairs.var()
                features[f'num_{num}_pair_entropy'] = stats.entropy(pairs + 1e-10)
                
                # Features de pares por janela temporal
                for window in [20, 50]:
                    if len(df) >= window:
                        recent_df = df.tail(window)
                        recent_cooc = self._calculate_recent_cooccurrence(recent_df, num)
                        features[f'num_{num}_recent_pair_strength_{window}'] = recent_cooc
            
            features['cooccurrence_matrix'] = cooccurrence_matrix
            
        except Exception as e:
            loto_logger.log_error(e, "Erro ao gerar features de pares")
            
        return features
    
    @performance_logger
    def _generate_frequency_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gera features avançadas de frequência"""
        features = {}
        
        try:
            for num in range(1, 101):
                col = f'num_{num}'
                data = df[col].values
                
                # Análise de gaps (intervalos entre aparições)
                gaps = self._calculate_gaps(data)
                if len(gaps) > 0:
                    features[f'{col}_avg_gap'] = np.mean(gaps)
                    features[f'{col}_gap_std'] = np.std(gaps)
                    features[f'{col}_max_gap'] = np.max(gaps)
                    features[f'{col}_min_gap'] = np.min(gaps)
                    features[f'{col}_current_gap'] = self._current_gap(data)
                    features[f'{col}_gap_trend'] = self._calculate_gap_trend(gaps)
                else:
                    features[f'{col}_avg_gap'] = 0
                    features[f'{col}_gap_std'] = 0
                    features[f'{col}_max_gap'] = 0
                    features[f'{col}_min_gap'] = 0
                    features[f'{col}_current_gap'] = len(data)
                    features[f'{col}_gap_trend'] = 0
                
                # Análise de streaks (sequências consecutivas)
                streaks = self._calculate_streaks(data)
                features[f'{col}_max_streak'] = max(streaks) if streaks else 0
                features[f'{col}_avg_streak'] = np.mean(streaks) if streaks else 0
                
                # Features de momentum
                features[f'{col}_momentum_5'] = self._calculate_momentum(data, 5)
                features[f'{col}_momentum_10'] = self._calculate_momentum(data, 10)
                
        except Exception as e:
            loto_logger.log_error(e, "Erro ao gerar features de frequência")
            
        return features
    
    @performance_logger
    def _generate_trend_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gera features de tendência"""
        features = {}
        
        try:
            for num in range(1, 101):
                col = f'num_{num}'
                data = df[col].values
                
                # Tendências em diferentes janelas
                for window in self.config.trend_windows:
                    if len(data) >= window:
                        recent_data = data[-window:]
                        
                        # Tendência linear
                        x = np.arange(len(recent_data))
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, recent_data)
                        
                        features[f'{col}_trend_slope_{window}'] = slope
                        features[f'{col}_trend_r2_{window}'] = r_value ** 2
                        features[f'{col}_trend_p_value_{window}'] = p_value
                        
                        # Tendência exponencial
                        exp_trend = self._calculate_exponential_trend(recent_data)
                        features[f'{col}_exp_trend_{window}'] = exp_trend
                        
                        # Média móvel
                        ma = np.convolve(recent_data, np.ones(min(5, window))/min(5, window), mode='valid')
                        if len(ma) > 0:
                            features[f'{col}_ma_trend_{window}'] = ma[-1] - ma[0] if len(ma) > 1 else 0
        
        except Exception as e:
            loto_logger.log_error(e, "Erro ao gerar features de tendência")
            
        return features
    
    @performance_logger
    def _generate_fourier_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gera features de análise de Fourier para detectar ciclos"""
        features = {}
        
        try:
            for num in range(1, 101):
                col = f'num_{num}'
                data = df[col].values
                
                if len(data) >= 20:  # Mínimo para análise de Fourier
                    # FFT
                    fft_data = fft(data)
                    power_spectrum = np.abs(fft_data) ** 2
                    
                    # Frequências dominantes
                    freqs = np.fft.fftfreq(len(data))
                    dominant_freq_idx = np.argsort(power_spectrum)[-self.config.fourier_components:]
                    
                    for i, idx in enumerate(dominant_freq_idx):
                        features[f'{col}_fourier_freq_{i}'] = freqs[idx]
                        features[f'{col}_fourier_power_{i}'] = power_spectrum[idx]
                    
                    # Features agregadas do espectro
                    features[f'{col}_spectral_centroid'] = np.sum(freqs * power_spectrum) / np.sum(power_spectrum)
                    features[f'{col}_spectral_spread'] = np.sqrt(np.sum(((freqs - features[f'{col}_spectral_centroid']) ** 2) * power_spectrum) / np.sum(power_spectrum))
                    features[f'{col}_spectral_entropy'] = stats.entropy(power_spectrum + 1e-10)
        
        except Exception as e:
            loto_logger.log_error(e, "Erro ao gerar features de Fourier")
            
        return features
    
    @performance_logger
    def _generate_gap_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gera features avançadas de análise de gaps"""
        features = {}
        
        try:
            for num in range(1, 101):
                col = f'num_{num}'
                data = df[col].values
                
                gaps = self._calculate_gaps(data)
                if len(gaps) > 0:
                    # Features estatísticas dos gaps
                    features[f'{col}_gap_median'] = np.median(gaps)
                    features[f'{col}_gap_mode'] = stats.mode(gaps, keepdims=True)[0][0] if len(gaps) > 1 else gaps[0]
                    features[f'{col}_gap_iqr'] = np.percentile(gaps, 75) - np.percentile(gaps, 25)
                    
                    # Distribuição dos gaps
                    features[f'{col}_gap_skew'] = stats.skew(gaps)
                    features[f'{col}_gap_kurtosis'] = stats.kurtosis(gaps)
                    
                    # Análise de periodicidade dos gaps
                    if len(gaps) > 3:
                        gap_autocorr = np.corrcoef(gaps[:-1], gaps[1:])[0, 1]
                        features[f'{col}_gap_autocorr'] = gap_autocorr if not np.isnan(gap_autocorr) else 0
                    
                    # Features de pressão (tendência a sair após gap longo)
                    current_gap = self._current_gap(data)
                    avg_gap = np.mean(gaps)
                    features[f'{col}_gap_pressure'] = max(0, (current_gap - avg_gap) / avg_gap) if avg_gap > 0 else 0
        
        except Exception as e:
            loto_logger.log_error(e, "Erro ao gerar features de gap")
            
        return features
    
    @performance_logger
    def _generate_pattern_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Gera features de análise de padrões"""
        features = {}
        
        try:
            # Análise de padrões por grupos de números
            groups = {
                'low': list(range(1, 21)),      # 1-20
                'mid_low': list(range(21, 41)), # 21-40
                'mid_high': list(range(41, 61)), # 41-60
                'high': list(range(61, 81)),    # 61-80
                'very_high': list(range(81, 101)) # 81-100
            }
            
            for group_name, nums in groups.items():
                group_data = df[[f'num_{num}' for num in nums]].sum(axis=1)
                
                features[f'group_{group_name}_freq'] = group_data.mean()
                features[f'group_{group_name}_std'] = group_data.std()
                features[f'group_{group_name}_trend'] = self._calculate_trend(group_data.values)
                
                # Análise de janelas para grupos
                for window in [10, 20]:
                    if len(group_data) >= window:
                        recent_data = group_data.tail(window)
                        features[f'group_{group_name}_recent_freq_{window}'] = recent_data.mean()
            
            # Análise de paridade (pares vs ímpares)
            even_nums = [num for num in range(2, 101, 2)]
            odd_nums = [num for num in range(1, 101, 2)]
            
            even_data = df[[f'num_{num}' for num in even_nums]].sum(axis=1)
            odd_data = df[[f'num_{num}' for num in odd_nums]].sum(axis=1)
            
            features['even_odd_ratio'] = even_data.mean() / odd_data.mean() if odd_data.mean() > 0 else 0
            features['even_freq'] = even_data.mean()
            features['odd_freq'] = odd_data.mean()
            
            # Análise de soma dos números sorteados
            total_sums = []
            for _, row in df.iterrows():
                contest_sum = sum(num for num in range(1, 101) if row[f'num_{num}'] == 1)
                total_sums.append(contest_sum)
            
            sum_series = pd.Series(total_sums)
            features['sum_mean'] = sum_series.mean()
            features['sum_std'] = sum_series.std()
            features['sum_trend'] = self._calculate_trend(sum_series.values)
            
            # Últimas somas para modelo
            features['recent_sums'] = sum_series.tail(20).values.tolist()
        
        except Exception as e:
            loto_logger.log_error(e, "Erro ao gerar features de padrão")
            
        return features
    
    # Métodos auxiliares
    def _calculate_trend(self, data: np.ndarray) -> float:
        """Calcula tendência linear dos dados"""
        if len(data) < 2:
            return 0.0
        x = np.arange(len(data))
        slope, _, _, _, _ = stats.linregress(x, data)
        return slope
    
    def _calculate_gaps(self, data: np.ndarray) -> List[int]:
        """Calcula gaps entre aparições"""
        gaps = []
        last_appearance = -1
        
        for i, val in enumerate(data):
            if val == 1:
                if last_appearance >= 0:
                    gaps.append(i - last_appearance - 1)
                last_appearance = i
        
        return gaps
    
    def _current_gap(self, data: np.ndarray) -> int:
        """Calcula gap atual desde última aparição"""
        for i in range(len(data) - 1, -1, -1):
            if data[i] == 1:
                return len(data) - 1 - i
        return len(data)
    
    def _calculate_streaks(self, data: np.ndarray) -> List[int]:
        """Calcula sequências consecutivas"""
        streaks = []
        current_streak = 0
        
        for val in data:
            if val == 1:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append(current_streak)
                current_streak = 0
        
        if current_streak > 0:
            streaks.append(current_streak)
        
        return streaks
    
    def _calculate_momentum(self, data: np.ndarray, window: int) -> float:
        """Calcula momentum (aceleração na frequência)"""
        if len(data) < window * 2:
            return 0.0
        
        recent = data[-window:].mean()
        previous = data[-window*2:-window].mean()
        
        return recent - previous
    
    def _calculate_seasonal_strength(self, data: np.ndarray) -> float:
        """Calcula força da sazonalidade"""
        if len(data) < 12:
            return 0.0
        
        # Decompor em componentes sazonais simples
        seasonal_period = min(12, len(data) // 4)
        if seasonal_period < 2:
            return 0.0
        
        seasonal_means = []
        for i in range(seasonal_period):
            indices = range(i, len(data), seasonal_period)
            if indices:
                seasonal_mean = np.mean([data[j] for j in indices if j < len(data)])
                seasonal_means.append(seasonal_mean)
        
        return np.std(seasonal_means) if len(seasonal_means) > 1 else 0.0
    
    def _calculate_exponential_trend(self, data: np.ndarray) -> float:
        """Calcula tendência exponencial"""
        if len(data) < 2:
            return 0.0
        
        try:
            # Usar decay exponencial
            weights = np.exp(-np.arange(len(data)) * 0.1)
            weights = weights / weights.sum()
            
            weighted_recent = np.sum(data * weights)
            simple_mean = data.mean()
            
            return weighted_recent - simple_mean
        except:
            return 0.0
    
    def _calculate_sequence_trends(self, sequences: np.ndarray) -> np.ndarray:
        """Calcula tendências para sequências"""
        trends = np.zeros(sequences.shape[1])  # Para cada número
        
        for num_idx in range(sequences.shape[1]):
            # Média das tendências de todas as sequências para este número
            num_trends = []
            for seq_idx in range(sequences.shape[0]):
                seq_data = sequences[seq_idx, num_idx, :]
                trend = self._calculate_trend(seq_data)
                num_trends.append(trend)
            
            trends[num_idx] = np.mean(num_trends)
        
        return trends
    
    def _calculate_recent_cooccurrence(self, recent_df: pd.DataFrame, target_num: int) -> float:
        """Calcula coocorrência recente para um número específico"""
        target_col = f'num_{target_num}'
        target_contests = recent_df[recent_df[target_col] == 1]
        
        if len(target_contests) == 0:
            return 0.0
        
        # Calcular média de coocorrências
        cooc_scores = []
        for _, contest in target_contests.iterrows():
            other_nums = [num for num in range(1, 101) if num != target_num and contest[f'num_{num}'] == 1]
            cooc_scores.append(len(other_nums))
        
        return np.mean(cooc_scores) if cooc_scores else 0.0
    
    def _calculate_gap_trend(self, gaps: List[int]) -> float:
        """Calcula tendência dos gaps"""
        if len(gaps) < 2:
            return 0.0
        return self._calculate_trend(np.array(gaps))
    
    def _cache_features(self, features: Dict[str, Any], contests: List[Dict[str, Any]]):
        """Cache das features geradas"""
        try:
            # Gerar chave de cache baseada nos concursos
            cache_key = self._generate_cache_key(contests)
            
            # Converter arrays numpy para listas para serialização
            cacheable_features = self._make_cacheable(features)
            
            self.cache.set(cache_key, cacheable_features, expire=3600 * 24)  # 24h
            self._feature_cache_keys.add(cache_key)
            
        except Exception as e:
            loto_logger.log_warning(f"Erro ao fazer cache das features: {e}")
    
    def _generate_cache_key(self, contests: List[Dict[str, Any]]) -> str:
        """Gera chave de cache baseada nos dados de entrada"""
        # Usar hash dos números dos concursos e configuração
        contest_numbers = [c['contest'] for c in contests[-100:]]  # Últimos 100
        cache_data = {
            'contests': contest_numbers,
            'config': {
                'lookback_windows': self.config.lookback_windows,
                'trend_windows': self.config.trend_windows,
                'fourier_components': self.config.fourier_components
            }
        }
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()
    
    def _make_cacheable(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Converte features para formato cacheável"""
        cacheable = {}
        
        for key, value in features.items():
            if isinstance(value, dict):
                cacheable[key] = self._make_cacheable(value)
            elif isinstance(value, np.ndarray):
                cacheable[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                cacheable[key] = float(value)
            else:
                cacheable[key] = value
        
        return cacheable
    
    @performance_logger
    def clear_cache(self):
        """Limpa cache de features"""
        try:
            for key in self._feature_cache_keys:
                if key in self.cache:
                    del self.cache[key]
            self._feature_cache_keys.clear()
            loto_logger.log_info("Cache de features limpo")
        except Exception as e:
            loto_logger.log_error(e, "Erro ao limpar cache de features")

# Instância global
feature_engineer = LotomaniaFeatureEngineer()