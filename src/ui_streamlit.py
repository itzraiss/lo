import json
import os

import numpy as np
import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Lotomania Predictor", layout="wide")

st.title("Lotomania – Previsão, Geração de Jogos e Simulação")

with st.sidebar:
    st.header("Ação")
    do_generate = st.checkbox("Gerar tickets", value=True)
    do_simulate = st.checkbox("Simular (Monte Carlo)", value=True)
    do_save = st.checkbox("Salvar no MongoDB", value=True)
    if st.button("Executar previsão"):
        with st.spinner("Executando..."):
            resp = requests.post(f"{API_URL}/predict", json={
                "generate": do_generate,
                "simulate": do_simulate,
                "save": do_save,
            })
            if resp.ok:
                st.session_state["last_result"] = resp.json()
            else:
                st.error(f"Falha: {resp.status_code} - {resp.text}")

# Load latest
if "last_result" not in st.session_state:
    try:
        r = requests.get(f"{API_URL}/latest")
        if r.ok and r.json().get("latest"):
            st.session_state["last_result"] = r.json()["latest"]
    except Exception:
        pass

res = st.session_state.get("last_result")

if res:
    st.subheader("Resumo da Última Execução")
    cols = st.columns(3)
    cols[0].metric("Model version", res.get("model_version", "-"))
    sim = res.get("simulations", {})
    cols[1].metric("p(≥17)", f"{sim.get('p_at_least_17', 0):.3%}")
    cols[2].metric("p(≥18)", f"{sim.get('p_at_least_18', 0):.3%}")

    st.subheader("Tickets (3 x 50 números)")
    tickets = res.get("tickets", [])
    for i, t in enumerate(tickets, start=1):
        st.write(f"Ticket {i}: {sorted(t)}")

    st.subheader("Probabilidades p_i (Top 30)")
    p_i = res.get("p_i", {})
    if p_i:
        series = pd.Series({int(k): float(v) for k, v in p_i.items()}).sort_values(ascending=False)[:30]
        st.dataframe(series.to_frame("p").style.format({"p": "{:.5f}"}))

    st.subheader("JSON bruto")
    st.code(json.dumps(res, indent=2, default=str)[:5000])
else:
    st.info("Nenhuma previsão disponível ainda. Use a ação na barra lateral para executar.")