# Monitor Inteligente de Qualidade do Ar (SP) + SHAP
Este projeto implementa um dashboard **Streamlit** para acompanhar e prever qualidade do ar em SP, com explicabilidade por **SHAP**.

---

## 🍃 Funcionamento (resumo)

### 1) Dados e histórico
- A aplicação busca dados na **API WAQI** (parque D. Pedro II).
- Ela registra o valor de **AQI** e salva/atualiza o arquivo `data/dataset.csv`.
- Na tela principal, o dashboard usa os **últimos `WINDOW_SIZE`** valores de AQI para formar a janela de entrada do modelo.

### 🌳 Modelo e escalonamento
- `model.keras` é carregado para prever o AQI da **próxima hora**.
- O `scaler.pkl` (MinMaxScaler) normaliza os valores.
- Se `scaler.pkl` não existir, o app recria o scaler a partir do dataset base `air_quality_sp.csv`.

### ⛰️ Explicabilidade com SHAP
- O app tenta montar um **background real** para o `shap.KernelExplainer` usando janelas construídas a partir de `air_quality_sp.csv`.
- Se houver falha (ex.: arquivo não encontrado), ele usa um background **aleatório** como fallback para não quebrar o dashboard.
- Ao clicar em **“Explicar esta previsão”**, o SHAP calcula a contribuição de cada hora passada na previsão.

---

## ⚙️ Como executar

1. Instale dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure o `.env` com:
   ```bash
   WAQI_TOKEN=seu_token
   ```

3. Rode:
   ```bash
   streamlit run dashboard_xai.py
   ```

---

## Arquivos principais
- `dashboard_xai.py` — dashboard + previsão + SHAP
- `model.keras` — modelo treinado
- `scaler.pkl` — normalizador (geralmente gerado durante o treinamento)
- `air_quality_sp.csv` — dataset base para background (SHAP) e fallback do scaler
- `data/dataset.csv` — histórico acumulado em runtime

