# dashboard_xai.py - COM SHAP KERNEL EXPLAINER

import streamlit as st
st.set_page_config(page_title="GAYA SENSO - Smart Air Quality Monitor", 
                   layout="wide"
                   )

# Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import shap
import os
import requests
import joblib
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AIR_QUALITY_CSV_PATH = os.path.join(BASE_DIR, 'air_quality_sp.csv')


# 1. CARREGAR MODELO E SCALER

@st.cache_resource
def load_model_and_scaler():
    model = load_model('model.keras')
    window_size = model.input_shape[1]

    try:
        scaler = joblib.load('scaler.pkl')
    except FileNotFoundError:
        st.warning("scaler.pkl não encontrado. Recriando a partir do CSV original...")
        df_raw = pd.read_csv(AIR_QUALITY_CSV_PATH)
        temp = df_raw['aqi'].values

        def df_to_X_y(data, w):
            X = []
            for i in range(len(data) - w):
                X.append([[x] for x in data[i:i+w]])
            return np.array(X)

        X_full = df_to_X_y(temp, window_size)
        X_flat = X_full.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaler.fit(X_flat)
        st.success("Scaler recriado com sucesso!")

    return model, scaler


model, scaler = load_model_and_scaler()
WINDOW_SIZE = model.input_shape[1]

# 2. CONFIGURAÇÕES DA API 

load_dotenv()
WAQI_TOKEN = os.getenv("WAQI_TOKEN")
if not WAQI_TOKEN:
    st.error("❌ Token WAQI não encontrado. Crie um arquivo .env com WAQI_TOKEN=seu_token")
    st.stop()

CITY_URL = "https://api.waqi.info/feed/brazil/sao-paulo/parque-d.pedro-ii/"
HISTORY_PATH = "data/dataset.csv"


# 3. FUNÇÕES DE COLETA E HISTÓRICO

def fetch_current_data():
    response = requests.get(CITY_URL, params={"token": WAQI_TOKEN}, timeout=10)
    data = response.json()
    if data.get("status") != "ok":
        raise Exception(f"Erro na API: {data.get('data')}")

    iaqi = data["data"]["iaqi"]
    row = {
        "timestamp": datetime.now(),
        "aqi": data["data"]["aqi"],
        "pm25": iaqi.get("pm25", {}).get("v"),
        "pm10": iaqi.get("pm10", {}).get("v"),
        "o3": iaqi.get("o3", {}).get("v"),
        "no2": iaqi.get("no2", {}).get("v"),
        "temp": iaqi.get("t", {}).get("v"),
        "humidity": iaqi.get("h", {}).get("v"),
    }
    return row


def update_history():
    os.makedirs("data", exist_ok=True)
    new_row = fetch_current_data()
    df_new = pd.DataFrame([new_row])

    if os.path.exists(HISTORY_PATH) and os.path.getsize(HISTORY_PATH) > 0:
        df_old = pd.read_csv(HISTORY_PATH, parse_dates=['timestamp'])
        df_all = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_all = df_new

    df_all.to_csv(HISTORY_PATH, index=False)

    aqi_series = df_all['aqi'].values[-WINDOW_SIZE:]
    if len(aqi_series) < WINDOW_SIZE:
        pad = [aqi_series[0]] * (WINDOW_SIZE - len(aqi_series))
        aqi_series = np.concatenate([pad, aqi_series])

    return aqi_series.reshape(-1, 1)


def preprocess_input(raw_seq):
    flat = raw_seq.reshape(-1, 1)
    scaled_flat = scaler.transform(flat)
    return scaled_flat.reshape(1, WINDOW_SIZE, 1)

# 4. EXPLAINER SHAP - KERNEL EXPLAINER

@st.cache_resource
def load_shap_explainer():
    # Construir background a partir do dataset original
    try:
        df_raw = pd.read_csv(AIR_QUALITY_CSV_PATH)
        temp = df_raw['aqi'].values

        def build_windows(data, w):
            X = []
            for i in range(len(data) - w):
                X.append([[x] for x in data[i:i+w]])
            return np.array(X)

        X_all = build_windows(temp, WINDOW_SIZE)
        # seleciona 100 janelas aleatórias
        idx = np.random.choice(len(X_all), min(100, len(X_all)), replace=False)
        background = X_all[idx]

        # escala o background
        bg_flat = background.reshape(-1, 1)
        bg_scaled_flat = scaler.transform(bg_flat)
        bg_scaled = bg_scaled_flat.reshape(background.shape)
        background_2d = bg_scaled.reshape(-1, WINDOW_SIZE)
    except Exception as e:
        st.warning(f"Erro ao carregar background real: {e}. Usando aleatório.")
        background_2d = np.random.randn(100, WINDOW_SIZE)

    def model_predict(data_2d):
        # data_2d: (n_samples, WINDOW_SIZE) - já escalado
        X_3d = data_2d.reshape(-1, WINDOW_SIZE, 1)
        return model.predict(X_3d, verbose=0).flatten()

    explainer = shap.KernelExplainer(model_predict, background_2d)
    return explainer


explainer = load_shap_explainer()

#-----------------------


# 5. DASHBOARD PRINCIPAL

st.title("Monitor Inteligente de Qualidade do Ar")
st.markdown("Previsão de AQI com LSTM + Explicabilidade (SHAP)")

if st.sidebar.button("🔄 Atualizar agora"):
    st.rerun()

with st.spinner("Buscando dados da estação Parque D. Pedro II..."):
    raw_seq = update_history()
    X_input = preprocess_input(raw_seq)
    prediction = model.predict(X_input, verbose=0)[0, 0]

col1, col2, col3 = st.columns(3)
col1.metric("AQI atual", f"{raw_seq[-1][0]:.0f}")
col2.metric("AQI previsto (próxima hora)", f"{prediction:.1f}")
risk = "ALTO" if prediction > 150 else "MÉDIO" if prediction > 100 else "BAIXO"
col3.metric("Nível de risco", risk)

st.subheader(f"Últimos {WINDOW_SIZE} valores de AQI")
timestamps = [datetime.now() - timedelta(hours=i) for i in range(WINDOW_SIZE-1, -1, -1)]
df_hist = pd.DataFrame({"Tempo": timestamps, "AQI": raw_seq.flatten()})
st.line_chart(df_hist.set_index("Tempo"))

if st.button("🔍 Explicar esta previsão"):
    with st.spinner("Calculando contribuições SHAP (pode levar até 30s)..."):
        instance_2d = X_input.reshape(1, WINDOW_SIZE)
        shap_values_2d = explainer.shap_values(instance_2d)
        if isinstance(shap_values_2d, list):
            shap_vals = shap_values_2d[0][0]
        else:
            shap_vals = shap_values_2d[0]
        input_vals = X_input[0, :, 0]

    df_shap = pd.DataFrame({
        "Horas atrás": [f"t-{i}" for i in range(WINDOW_SIZE-1, -1, -1)],
        "AQI (escalado)": input_vals,
        "Contribuição SHAP": shap_vals
    })
    st.dataframe(df_shap)

    fig, ax = plt.subplots()
    ax.barh(df_shap["Horas atrás"], df_shap["Contribuição SHAP"], color='skyblue')
    ax.set_xlabel("Impacto SHAP (positivo = aumenta previsão)")
    ax.set_title("Influência de cada hora passada na previsão")
    st.pyplot(fig)

    pos = df_shap[df_shap["Contribuição SHAP"] > 0]
    neg = df_shap[df_shap["Contribuição SHAP"] < 0]
    if not pos.empty:
        st.info(f"📈 Aumentaram o risco: {', '.join(pos['Horas atrás'].values)}")
    if not neg.empty:
        st.info(f"📉 Diminuíram o risco: {', '.join(neg['Horas atrás'].values)}")


# streamlit run dashboard_xai.py

