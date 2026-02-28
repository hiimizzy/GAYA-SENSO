import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

# CONFIGURAÇÃO DA PÁGINA

st.set_page_config(
    page_title="Qualidade do Ar",
    layout="centered"
)


# CARREGAR VARIÁVEIS DE AMBIENTE

load_dotenv()

WAQI_TOKEN = os.getenv("WAQI_TOKEN")

if not WAQI_TOKEN:
    st.error("Token WAQI não encontrado. Verifique o arquivo .env")
    st.stop()  


# CONFIGURAÇÕES

CITY_URL = "https://api.waqi.info/feed/@8490/"  # Parque D. Pedro II
DATA_PATH = "data/dataset.csv"


# BUSCAR DADOS DA API

def fetch_air_quality():
    url = f"{CITY_URL}?token={WAQI_TOKEN}"
    response = requests.get(url, timeout=10)
    data = response.json()

    if data.get("status") != "ok":
        st.error("Erro ao acessar a API WAQI")
        return None

    iaqi = data["data"]["iaqi"]

    row = {
        "timestamp": datetime.now(),
        "aqi": data["data"]["aqi"],
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
        "co": iaqi.get("co", {}).get("v"),
        "so2": iaqi.get("so2", {}).get("v"),
        "temp": iaqi.get("t", {}).get("v"),
        "humidity": iaqi.get("h", {}).get("v"),
    }

    return pd.DataFrame([row])

# SALVAR DADOS NO CSV

def save_data(df_new):
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    # Arquivo não existe ou está vazio
    if not os.path.exists(DATA_PATH) or os.stat(DATA_PATH).st_size == 0:
        df_new.to_csv(DATA_PATH, index=False)
        return df_new

    # Caso normal: concatena histórico
    df_old = pd.read_csv(DATA_PATH)
    df_all = pd.concat([df_old, df_new], ignore_index=True)
    df_all.to_csv(DATA_PATH, index=False)

    return df_all


# INTERFACE STREAMLIT

st.title("🌫️>⩊< Qualidade do Ar — Parque D. Pedro II (SP)")

if st.button("📡 Coletar dados agora"):
    df_new = fetch_air_quality()

    if df_new is not None:
        df_all = save_data(df_new)
        st.success("Dados coletados e salvos com sucesso!")
        st.metric("AQI Atual", int(df_new["aqi"].iloc[0]))


# VISUALIZAR OS DADOS

if os.path.exists(DATA_PATH) and os.stat(DATA_PATH).st_size > 0:
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    st.subheader("📊 Histórico de AQI")
    st.line_chart(df.set_index("timestamp")["aqi"])

    st.subheader("📄 Dados brutos")
    st.dataframe(df)
else:
    st.info("Nenhum dado coletado ainda.")
    
# "c:\Users\arauj\OneDrive\Área de Trabalho\air_quality_sp\air_quality"
# 
# cd air_quality
# streamlit run app.py