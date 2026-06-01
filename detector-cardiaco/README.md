#  Detector de Risco de Ataque Cardíaco

Rede Neural profunda com TensorFlow/Keras para classificar risco cardíaco baseado em dados clínicos.

##  O que o projeto faz

- Gera dataset clínico simulado baseado no Cleveland Heart Disease Dataset com **Pandas + NumPy**
- EDA completa com análise de correlações e estatísticas por grupo
- Feature engineering: pressão por idade, reserva cardíaca, índice de risco
- Rede Neural com **TensorFlow/Keras**:
  - 4 camadas densas com BatchNormalization + Dropout
  - Regularização L2
  - EarlyStopping e ReduceLROnPlateau
  - Class weights para dados desbalanceados
- Comparação com baselines do **scikit-learn** (Regressão Logística + Random Forest)
- Predição individual de pacientes com nível de risco
- Salva o modelo treinado em `.keras`
- Painel de visualizações com **Matplotlib** e **Seaborn**:
  - Loss e AUC por época
  - Curva ROC com área preenchida
  - Matriz de confusão
  - Distribuição das probabilidades por classe
  - Correlações com risco cardíaco

##  Como rodar

```bash
pip install -r requirements.txt
python detector_cardiaco.py
```

##  Arquivos gerados

- `detector_cardiaco.png` — painel visual completo
- `dataset_cardiaco.csv` — dataset processado
- `modelo_cardiaco.keras` — modelo salvo

##  Bibliotecas

| Biblioteca | Uso |
|---|---|
| **TensorFlow/Keras** | Rede Neural profunda |
| **scikit-learn** | Baselines e pré-processamento |
| **Pandas** | Manipulação e EDA |
| **NumPy** | Cálculos numéricos |
| **Matplotlib** | Curvas de treinamento e ROC |
| **Seaborn** | Heatmap e KDE |

##  Aviso
Projeto educacional. Consulte sempre um médico para diagnósticos reais.
