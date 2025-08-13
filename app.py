import streamlit as st
import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

# Local imports
from config import config
from database.mongodb_client import mongodb_client
from data_ingestion.data_loader import LotomaniaDataLoader, update_from_api_async
from features.feature_engineering import feature_engineer
from models.ensemble_models import ensemble_model
from optimization.ticket_optimizer import ticket_optimizer
from simulation.monte_carlo import monte_carlo_simulator
from utils.logger import loto_logger
from sample_data import SAMPLE_PREDICTION, generate_sample_contests

# Configurar página
st.set_page_config(
    page_title="Lotomania AI - Sistema Automatizado",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estado global da aplicação
if 'app_state' not in st.session_state:
    st.session_state.app_state = {
        'system_running': False,
        'last_prediction': None,
        'last_update': None,
        'cycle_count': 0,
        'system_status': 'Initializing',
        'background_thread': None,
        'auto_mode': True,
        'connected_to_db': False,
        'models_trained': False,
        'demo_mode': False
    }

class LotomaniaApp:
    """Aplicação principal do sistema Lotomania"""
    
    def __init__(self):
        self.running = False
        self.prediction_cycle_interval = config.api.update_interval_hours * 3600
        self.last_cycle_time = 0
        
    async def initialize_system(self):
        """Inicializa todos os componentes do sistema"""
        try:
            st.session_state.app_state['system_status'] = 'Connecting to database...'
            
            # Conectar ao MongoDB
            if not mongodb_client.is_connected:
                await mongodb_client.connect()
            st.session_state.app_state['connected_to_db'] = True
            
            st.session_state.app_state['system_status'] = 'Loading historical data...'
            
            # Verificar se há dados históricos
            contests = await mongodb_client.get_contests(limit=10)
            if len(contests) < 100:
                # Tentar carregar dados históricos se houver arquivo
                if os.path.exists('results.xlsx'):
                    async with LotomaniaDataLoader() as loader:
                        await loader.full_historical_load('results.xlsx')
            
            st.session_state.app_state['system_status'] = 'Training models...'
            
            # Treinar modelos se necessário
            if not ensemble_model.is_trained:
                await self.train_models_if_needed()
            
            st.session_state.app_state['models_trained'] = ensemble_model.is_trained
            st.session_state.app_state['system_status'] = 'System ready'
            
            loto_logger.log_info("Sistema inicializado com sucesso")
            return True
            
        except Exception as e:
            error_msg = f"Erro na inicialização: {str(e)}"
            st.session_state.app_state['system_status'] = error_msg
            loto_logger.log_error(e, "Erro na inicialização do sistema")
            return False
    
    async def train_models_if_needed(self):
        """Treina modelos se necessário"""
        try:
            contests = await mongodb_client.get_contests()
            
            if len(contests) < config.model.min_training_contests:
                loto_logger.log_warning(f"Dados insuficientes para treinamento: {len(contests)} < {config.model.min_training_contests}")
                return False
            
            # Gerar features
            features = await feature_engineer.engineer_features(contests)
            
            # Treinar ensemble
            metrics = await ensemble_model.train_ensemble(features, contests)
            
            loto_logger.log_info(f"Modelos treinados com sucesso: {metrics}")
            return True
            
        except Exception as e:
            loto_logger.log_error(e, "Erro no treinamento de modelos")
            return False
    
    async def run_prediction_cycle(self):
        """Executa um ciclo completo de predição"""
        try:
            cycle_start = time.time()
            cycle_data = {
                'timestamp': datetime.now().isoformat(),
                'cycle_number': st.session_state.app_state['cycle_count'] + 1
            }
            
            # 1. Atualizar dados da API
            st.session_state.app_state['system_status'] = 'Updating data from API...'
            async with LotomaniaDataLoader() as loader:
                updated = await loader.incremental_update()
            
            if updated:
                loto_logger.log_info("Novos dados encontrados, retreinando modelos")
                await self.train_models_if_needed()
            
            # 2. Buscar dados mais recentes
            contests = await mongodb_client.get_contests()
            if not contests:
                loto_logger.log_warning("Nenhum dado de concurso encontrado")
                return None
            
            # 3. Gerar features
            st.session_state.app_state['system_status'] = 'Generating features...'
            features = await feature_engineer.engineer_features(contests)
            
            # 4. Gerar probabilidades com ensemble
            st.session_state.app_state['system_status'] = 'Generating probabilities...'
            if not ensemble_model.is_trained:
                loto_logger.log_warning("Modelos não treinados, usando estimativas básicas")
                probabilities = self._generate_basic_probabilities()
            else:
                probabilities = await ensemble_model.predict_probabilities(features)
            
            # 5. Otimizar tickets
            st.session_state.app_state['system_status'] = 'Optimizing tickets...'
            optimization_result = await ticket_optimizer.optimize_tickets(probabilities)
            
            if not optimization_result.get('success', False):
                loto_logger.log_error("Otimização de tickets falhou", "prediction_cycle")
                return None
            
            tickets = optimization_result['tickets']
            
            # 6. Simular probabilidades com Monte Carlo
            st.session_state.app_state['system_status'] = 'Running Monte Carlo simulation...'
            simulation_result = await monte_carlo_simulator.simulate_ticket_probabilities(
                tickets, probabilities
            )
            
            # 7. Verificar critérios de confiança
            confidence_metrics = simulation_result.get('confidence_metrics', {})
            passes_criteria = confidence_metrics.get('passes_criteria', {})
            
            # 8. Preparar resultado final
            prediction_result = {
                'generated_at': datetime.utcnow(),
                'model_version': ensemble_model.model_version or 'unknown',
                'probabilities': probabilities,
                'tickets': tickets,
                'optimization_result': optimization_result,
                'simulation_result': simulation_result,
                'confidence_passed': passes_criteria.get('any_criterion', False),
                'recommendation': confidence_metrics.get('recommendation', {}),
                'cycle_duration': time.time() - cycle_start
            }
            
            # 9. Salvar no banco
            prediction_id = await mongodb_client.insert_prediction(prediction_result)
            
            # 10. Atualizar estado da aplicação
            st.session_state.app_state['last_prediction'] = prediction_result
            st.session_state.app_state['last_update'] = datetime.now()
            st.session_state.app_state['cycle_count'] += 1
            st.session_state.app_state['system_status'] = 'Prediction cycle completed'
            
            # 11. Log do ciclo
            loto_logger.log_prediction_cycle({
                'prediction_id': prediction_id,
                'duration': prediction_result['cycle_duration'],
                'confidence_passed': prediction_result['confidence_passed'],
                'recommendation': prediction_result['recommendation'].get('action', 'UNKNOWN')
            })
            
            self.last_cycle_time = time.time()
            
            return prediction_result
            
        except Exception as e:
            error_msg = f"Erro no ciclo de predição: {str(e)}"
            st.session_state.app_state['system_status'] = error_msg
            loto_logger.log_error(e, "Erro no ciclo de predição")
            return None
    
    def _generate_basic_probabilities(self) -> Dict[str, float]:
        """Gera probabilidades básicas quando modelos não estão disponíveis"""
        import random
        probabilities = {}
        for i in range(1, 101):
            # Probabilidade base com alguma variação
            base_prob = 0.2  # 20% base para cada número
            variation = random.uniform(-0.05, 0.05)
            probabilities[str(i)] = max(0.01, base_prob + variation)
        
        return probabilities
    
    async def continuous_background_process(self):
        """Processo contínuo em background"""
        while self.running:
            try:
                current_time = time.time()
                
                # Verificar se é hora de executar novo ciclo
                if (current_time - self.last_cycle_time) >= self.prediction_cycle_interval:
                    await self.run_prediction_cycle()
                
                # Cleanup automático do MongoDB
                if st.session_state.app_state['cycle_count'] % 10 == 0:
                    await mongodb_client.cleanup_old_data()
                
                # Aguardar antes da próxima verificação
                await asyncio.sleep(60)  # Verificar a cada minuto
                
            except Exception as e:
                loto_logger.log_error(e, "Erro no processo contínuo")
                await asyncio.sleep(300)  # Aguardar 5 minutos em caso de erro
    
    def start_background_process(self):
        """Inicia processo em background"""
        if not self.running:
            self.running = True
            
            def run_async_loop():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self.continuous_background_process())
            
            thread = threading.Thread(target=run_async_loop, daemon=True)
            thread.start()
            st.session_state.app_state['background_thread'] = thread
            st.session_state.app_state['system_running'] = True
    
    def stop_background_process(self):
        """Para processo em background"""
        self.running = False
        st.session_state.app_state['system_running'] = False

# Instância global da aplicação
app = LotomaniaApp()

def main():
    """Função principal da aplicação Streamlit"""
    st.title("🎯 Lotomania AI - Sistema Automatizado")
    st.markdown("### Sistema inteligente para análise e predição da Lotomania")
    
    # Sidebar com controles
    with st.sidebar:
        st.header("⚙️ Controles do Sistema")
        
        # Status do sistema
        status_color = "green" if st.session_state.app_state['system_running'] else "red"
        st.markdown(f"**Status:** :{status_color}[{st.session_state.app_state['system_status']}]")
        
        # Botões de controle
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🚀 Inicializar", disabled=st.session_state.app_state['system_running']):
                with st.spinner("Inicializando sistema..."):
                    success = asyncio.run(app.initialize_system())
                    if success:
                        app.start_background_process()
                        st.success("Sistema iniciado!")
                        st.rerun()
        
        with col2:
            if st.button("⏹️ Parar", disabled=not st.session_state.app_state['system_running']):
                app.stop_background_process()
                st.warning("Sistema parado!")
                st.rerun()
        
        # Forçar novo ciclo
        if st.button("🔄 Executar Ciclo", disabled=not st.session_state.app_state['system_running']):
            with st.spinner("Executando ciclo de predição..."):
                result = asyncio.run(app.run_prediction_cycle())
                if result:
                    st.success("Ciclo executado com sucesso!")
                else:
                    st.error("Erro no ciclo de predição")
                st.rerun()
        
        # Modo Demo
        st.header("🎬 Modo Demonstração")
        if st.button("🎯 Ver Demo", help="Mostra exemplo de funcionamento do sistema"):
            st.session_state.app_state['demo_mode'] = True
            st.session_state.app_state['last_prediction'] = SAMPLE_PREDICTION
            st.session_state.app_state['last_update'] = datetime.now()
            st.success("Modo demo ativado!")
            st.rerun()
        
        if st.button("❌ Sair do Demo", disabled=not st.session_state.app_state['demo_mode']):
            st.session_state.app_state['demo_mode'] = False
            st.session_state.app_state['last_prediction'] = None
            st.info("Modo demo desativado!")
            st.rerun()
        
        # Configurações
        st.header("📊 Informações")
        st.metric("Ciclos Executados", st.session_state.app_state['cycle_count'])
        
        if st.session_state.app_state['last_update']:
            st.metric("Última Atualização", 
                     st.session_state.app_state['last_update'].strftime("%H:%M:%S"))
        
        # Status dos componentes
        st.header("🔧 Status dos Componentes")
        st.write("💾 Database:", "✅ Conectado" if st.session_state.app_state['connected_to_db'] else "❌ Desconectado")
        st.write("🤖 Modelos:", "✅ Treinados" if st.session_state.app_state['models_trained'] else "❌ Não treinados")
    
    # Área principal
    if st.session_state.app_state['demo_mode'] or st.session_state.app_state['last_prediction']:
        # Mostrar resultados (demo ou real)
        if st.session_state.app_state['demo_mode']:
            st.info("🎬 **MODO DEMONSTRAÇÃO** - Este é um exemplo de como o sistema funciona")
        show_prediction_results()
        
    elif not st.session_state.app_state['system_running']:
        st.info("🚀 Clique em 'Inicializar' na barra lateral para começar o sistema automatizado!")
        
        # Instruções
        st.markdown("""
        ## 📋 Como usar o sistema:
        
        1. **Demonstração**: Clique em "🎯 Ver Demo" na barra lateral para ver exemplo de funcionamento
        2. **Inicializar**: Clique no botão 'Inicializar' para conectar ao banco de dados e treinar os modelos
        3. **Funcionamento Automático**: O sistema irá automaticamente:
           - Verificar novos concursos da API da Caixa a cada 6 horas
           - Retreinar modelos quando necessário
           - Gerar predições com 90-95% de confiança
           - Otimizar 3 jogos de 50 números cada
           - Simular probabilidades de acerto
        4. **Monitoramento**: Acompanhe o status em tempo real na barra lateral
        
        ## 🎯 Critérios de Confiança:
        - **18+ acertos**: ≥ 2% de probabilidade
        - **17+ acertos**: ≥ 10% de probabilidade  
        - **16+ acertos**: ≥ 50% de probabilidade
        
        O sistema só emitirá recomendações quando pelo menos um critério for atendido.
        
        ## 🛠️ Configuração Necessária:
        
        Para usar o sistema completo, você precisa:
        1. **MongoDB**: Configure a variável `MONGODB_URI` nos secrets do HF Spaces
        2. **Dados Históricos**: Faça upload do arquivo `results.xlsx` com dados da Lotomania
        3. **Recursos**: O sistema usa otimizações para funcionar no tier gratuito (2 vCPU + 16GB RAM)
        """)
        
    else:
        # Sistema em funcionamento - aguardando resultados
        st.info("⏳ Sistema em funcionamento. Aguardando primeiro ciclo de predição...")
        
        # Progress bar simulado
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(100):
            progress_bar.progress(i + 1)
            status_text.text(f"Processando... {i+1}%")
            time.sleep(0.1)

def show_prediction_results():
    """Exibe resultados da última predição"""
    prediction = st.session_state.app_state['last_prediction']
    
    # Header com informações principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        recommendation = prediction.get('recommendation', {})
        action = recommendation.get('action', 'UNKNOWN')
        color = "green" if action == "PLAY" else "red" if action == "WAIT" else "gray"
        st.metric("Recomendação", action, delta=None)
    
    with col2:
        confidence = recommendation.get('confidence', 'UNKNOWN')
        st.metric("Confiança", confidence)
    
    with col3:
        sim_result = prediction.get('simulation_result', {})
        prob_18 = sim_result.get('combined_probabilities', {}).get('at_least_18', 0)
        st.metric("P(≥18 acertos)", f"{prob_18:.4f}")
    
    with col4:
        prob_17 = sim_result.get('combined_probabilities', {}).get('at_least_17', 0)
        st.metric("P(≥17 acertos)", f"{prob_17:.4f}")
    
    # Tabs para diferentes visualizações
    tab1, tab2, tab3, tab4 = st.tabs(["🎯 Jogos Recomendados", "📊 Análise", "📈 Probabilidades", "🔍 Detalhes"])
    
    with tab1:
        show_recommended_games(prediction)
    
    with tab2:
        show_analysis_charts(prediction)
    
    with tab3:
        show_probability_charts(prediction)
    
    with tab4:
        show_detailed_info(prediction)

def show_recommended_games(prediction):
    """Mostra os jogos recomendados"""
    tickets = prediction.get('tickets', [])
    sim_result = prediction.get('simulation_result', {})
    
    st.header("🎯 Jogos Otimizados")
    
    for i, ticket in enumerate(tickets):
        with st.expander(f"🎫 Jogo {i+1} - {len(ticket)} números", expanded=True):
            # Organizar números em grade
            cols = st.columns(10)
            for j, num in enumerate(sorted(ticket)):
                with cols[j % 10]:
                    st.button(str(num), key=f"ticket_{i}_num_{j}", disabled=True)
            
            # Estatísticas do ticket
            individual_tickets = sim_result.get('individual_tickets', [])
            if i < len(individual_tickets):
                ticket_stats = individual_tickets[i]
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    hit_probs = ticket_stats.get('hit_probabilities', {})
                    st.metric("P(≥18)", f"{hit_probs.get(18, 0):.4f}")
                
                with col2:
                    st.metric("P(≥17)", f"{hit_probs.get(17, 0):.4f}")
                
                with col3:
                    stats = ticket_stats.get('statistics', {})
                    st.metric("Acertos Esperados", f"{stats.get('mean_hits', 0):.1f}")

def show_analysis_charts(prediction):
    """Mostra gráficos de análise"""
    st.header("📊 Análise Estatística")
    
    # Gráfico de distribuição de probabilidades
    probabilities = prediction.get('probabilities', {})
    if probabilities:
        nums = [int(k) for k in probabilities.keys()]
        probs = [float(v) for v in probabilities.values()]
        
        fig = px.bar(x=nums, y=probs, title="Distribuição de Probabilidades por Número")
        fig.update_layout(xaxis_title="Número", yaxis_title="Probabilidade")
        st.plotly_chart(fig, use_container_width=True)
    
    # Análise de overlap
    optimization_result = prediction.get('optimization_result', {})
    overlap_analysis = optimization_result.get('overlap_analysis', {})
    
    if overlap_analysis:
        st.subheader("Análise de Sobreposição entre Jogos")
        overlaps = overlap_analysis.get('pairwise_overlaps', {})
        
        if overlaps:
            pairs = list(overlaps.keys())
            values = list(overlaps.values())
            
            fig = px.bar(x=pairs, y=values, title="Números em Comum entre Jogos")
            fig.update_layout(xaxis_title="Par de Jogos", yaxis_title="Números em Comum")
            st.plotly_chart(fig, use_container_width=True)

def show_probability_charts(prediction):
    """Mostra gráficos de probabilidades"""
    st.header("📈 Análise de Probabilidades de Acerto")
    
    sim_result = prediction.get('simulation_result', {})
    combined_probs = sim_result.get('combined_probabilities', {})
    
    if combined_probs:
        # Gráfico de barras das probabilidades combinadas
        hits = [int(k.split('_')[-1]) for k in combined_probs.keys()]
        probs = [float(v) for v in combined_probs.values()]
        
        fig = px.bar(x=hits, y=probs, title="Probabilidade de Acertar pelo Menos N Números")
        fig.update_layout(xaxis_title="Número de Acertos", yaxis_title="Probabilidade")
        
        # Adicionar linhas de referência dos critérios
        fig.add_hline(y=config.model.confidence_threshold_18, line_dash="dash", 
                     annotation_text="Critério 18 acertos")
        fig.add_hline(y=config.model.confidence_threshold_17, line_dash="dash", 
                     annotation_text="Critério 17 acertos")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Distribuição de acertos por ticket
    individual_tickets = sim_result.get('individual_tickets', [])
    if individual_tickets:
        st.subheader("Probabilidades Individuais por Jogo")
        
        ticket_data = []
        for ticket in individual_tickets:
            ticket_num = ticket['ticket_number']
            hit_probs = ticket['hit_probabilities']
            
            for hits, prob in hit_probs.items():
                ticket_data.append({
                    'Jogo': f"Jogo {ticket_num}",
                    'Acertos': hits,
                    'Probabilidade': prob
                })
        
        if ticket_data:
            df = pd.DataFrame(ticket_data)
            fig = px.bar(df, x='Acertos', y='Probabilidade', color='Jogo', 
                        title="Probabilidades por Jogo e Número de Acertos",
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)

def show_detailed_info(prediction):
    """Mostra informações detalhadas"""
    st.header("🔍 Informações Detalhadas")
    
    # Informações gerais
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Metadados")
        st.write("**Gerado em:**", prediction.get('generated_at', 'N/A'))
        st.write("**Versão do Modelo:**", prediction.get('model_version', 'N/A'))
        st.write("**Duração do Ciclo:**", f"{prediction.get('cycle_duration', 0):.2f}s")
        
        confidence_metrics = prediction.get('simulation_result', {}).get('confidence_metrics', {})
        passes_criteria = confidence_metrics.get('passes_criteria', {})
        
        st.write("**Critérios Atendidos:**")
        for criterion, passed in passes_criteria.items():
            emoji = "✅" if passed else "❌"
            st.write(f"  {emoji} {criterion}")
    
    with col2:
        st.subheader("🎲 Configuração da Simulação")
        sim_config = prediction.get('simulation_result', {}).get('simulation_config', {})
        
        st.write("**Simulações Monte Carlo:**", sim_config.get('num_simulations', 'N/A'))
        st.write("**Processamento Paralelo:**", sim_config.get('parallel_processing', 'N/A'))
        
        quality = prediction.get('simulation_result', {}).get('simulation_quality', {})
        st.write("**Qualidade da Simulação:**", quality.get('convergence_quality', 'N/A'))
    
    # Dados brutos (expansível)
    with st.expander("📄 Dados Brutos (JSON)"):
        st.json(prediction)

# Auto-refresh para atualizar a interface
if st.session_state.app_state['system_running']:
    time.sleep(10)  # Atualizar a cada 10 segundos
    st.rerun()

if __name__ == "__main__":
    main()