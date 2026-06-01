

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, callbacks, regularizers

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

import warnings
warnings.filterwarnings("ignore")
tf.get_logger().setLevel("ERROR")

sns.set_theme(style="darkgrid")
plt.rcParams.update({
    "figure.facecolor": "#0d1424",
    "axes.facecolor":   "#111827",
    "axes.labelcolor":  "#94a3b8",
    "axes.edgecolor":   "#1e293b",
    "xtick.color":      "#64748b",
    "ytick.color":      "#64748b",
    "text.color":       "#f1f5f9",
    "grid.color":       "#1e293b",
})
CORES = ["#3b82f6","#06b6d4","#10b981","#f59e0b","#8b5cf6","#ec4899"]

print(f" TensorFlow {tf.__version__} carregado!")

def gerar_dataset(n: int = 1000) -> pd.DataFrame:
    
    np.random.seed(42)

    df = pd.DataFrame({
        "idade":          np.random.randint(29, 78, size=n),
        "sexo":           np.random.choice([0, 1], size=n, p=[0.32, 0.68]),
        "tipo_dor":       np.random.choice([0, 1, 2, 3], size=n),
        "pressao":        np.random.randint(94, 201, size=n),
        "colesterol":     np.random.randint(126, 565, size=n),
        "glicemia":       np.random.choice([0, 1], size=n, p=[0.86, 0.14]),
        "ecg":            np.random.choice([0, 1, 2], size=n, p=[0.50, 0.28, 0.22]),
        "fc_maxima":      np.random.randint(71, 203, size=n),
        "angina":         np.random.choice([0, 1], size=n, p=[0.67, 0.33]),
        "depressao_st":   np.round(np.random.uniform(0, 6.2, size=n), 1),
        "inclinacao_st":  np.random.choice([0, 1, 2], size=n),
        "vasos_coloridos":np.random.choice([0, 1, 2, 3], size=n, p=[0.58, 0.22, 0.12, 0.08]),
        "talassemia":     np.random.choice([1, 2, 3], size=n, p=[0.06, 0.54, 0.40]),
    })

    score = (
          0.04  * df["idade"]
        + 0.30  * df["sexo"]
        + 0.25  * (3 - df["tipo_dor"])
        + 0.008 * df["pressao"]
        + 0.002 * df["colesterol"]
        + 0.35  * df["glicemia"]
        - 0.012 * df["fc_maxima"]
        + 0.30  * df["angina"]
        + 0.25  * df["depressao_st"]
        + 0.20  * df["vasos_coloridos"]
        + 0.15  * df["ecg"]
        - 5.0
    )

    prob = 1 / (1 + np.exp(-score))
    df["risco"] = (np.random.uniform(size=n) < prob).astype(int)

    # Feature engineering
    df["pressao_por_idade"]   = (df["pressao"] / df["idade"]).round(2)
    df["reserva_cardiaca"]    = df["fc_maxima"] - df["idade"]
    df["indice_risco_total"]  = (df["depressao_st"] * df["vasos_coloridos"] + df["angina"]).round(2)

    return df

def eda(df: pd.DataFrame):
    sep = "=" * 60
    print(f"\n{sep}")
    print("    DETECTOR CARDÍACO — ANÁLISE EXPLORATÓRIA")
    print(f"{sep}")
    print(f"\n INFO DO DATASET")
    print(f"  Shape         : {df.shape}")
    print(f"  Nulos         : {df.isnull().sum().sum()}")
    print(f"  Taxa de risco : {df['risco'].mean():.1%} ({df['risco'].sum()} pacientes)")

    print(f"\n   ESTATÍSTICAS POR GRUPO (Pandas groupby)")
    cols = ["idade","pressao","colesterol","fc_maxima","depressao_st"]
    agg  = df.groupby("risco")[cols].mean().round(2)
    agg.index = ["Baixo Risco","Alto Risco"]
    print(agg.to_string())

    print(f"\n  CORRELAÇÃO COM RISCO (NumPy)")
    print(f"  {'─'*52}")
    cols_num = ["idade","pressao","colesterol","fc_maxima","depressao_st",
                "vasos_coloridos","angina","reserva_cardiaca"]
    corrs = df[cols_num + ["risco"]].corr()["risco"].drop("risco").sort_values()
    for feat, corr in corrs.items():
        sinal = "+" if corr > 0 else ""
        barra = "█" * int(abs(corr) * 30)
        print(f"  {feat:<22} {sinal}{corr:.3f}  {barra}")
    print(f"\n{sep}")

def construir_modelo(input_dim: int) -> keras.Model:
    
    modelo = keras.Sequential([
        layers.Input(shape=(input_dim,)),

        layers.Dense(128, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.3),

        layers.Dense(64, kernel_regularizer=regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.2),

        layers.Dense(32, activation="relu"),
        layers.Dropout(0.1),

        layers.Dense(16, activation="relu"),

        layers.Dense(1, activation="sigmoid"),
    ], name="HeartRiskNet")

    modelo.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall")],
    )
    return modelo

def treinar(df: pd.DataFrame):
    """Pipeline completo de treinamento."""

    features = [c for c in df.columns if c != "risco"]
    X = df[features].values
    y = df["risco"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    print(f"\n   SPLIT DOS DADOS")
    print(f"  Treino     : {len(X_train)}")
    print(f"  Validação  : {len(X_val)}")
    print(f"  Teste      : {len(X_test)}")

    modelo = construir_modelo(X_train.shape[1])
    print(f"\n   ARQUITETURA DA REDE NEURAL (TensorFlow {tf.__version__})")
    modelo.summary()

    cbs = [
        callbacks.EarlyStopping(monitor="val_auc", patience=15,
                                restore_best_weights=True, mode="max"),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                   patience=8, min_lr=1e-6),
    ]
    

    print(f"\n  TREINANDO...")
    history = modelo.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=cbs,
        verbose=0,
        class_weight={0: 1.0, 1: len(y_train[y_train==0])/len(y_train[y_train==1])},
    )
    print(f"  → Treinamento concluído em {len(history.history['loss'])} épocas")

    return modelo, history, X_test, y_test, X_train, y_train, scaler, features

def avaliar(modelo, history, X_test, y_test, X_train, y_train):
    """Avalia o modelo e compara com baselines."""
    y_proba = modelo.predict(X_test, verbose=0).flatten()
    y_pred  = (y_proba >= 0.5).astype(int)

    sep = "=" * 60
    print(f"\n{sep}")
    print("  RESULTADO — REDE NEURAL (TensorFlow)")
    print(f"{sep}")
    print(f"  Acurácia  : {accuracy_score(y_test, y_pred):.1%}")
    print(f"  Precisão  : {precision_score(y_test, y_pred):.1%}")
    print(f"  Recall    : {recall_score(y_test, y_pred):.1%}")
    print(f"  F1-Score  : {f1_score(y_test, y_pred):.1%}")
    print(f"  AUC-ROC   : {roc_auc_score(y_test, y_proba):.3f}")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\n  MATRIZ DE CONFUSÃO")
    print(f"  TP: {cm[1,1]:>4}  FP: {cm[0,1]:>4}")
    print(f"  FN: {cm[1,0]:>4}  TN: {cm[0,0]:>4}")

    print(f"\n  CLASSIFICATION REPORT")
    print(classification_report(y_test, y_pred,
          target_names=["Baixo Risco","Alto Risco"]))

    # Baselines
    print(f"\n  COMPARAÇÃO COM BASELINES (scikit-learn)")
    print(f"  {'─'*52}")
    baselines = {
        "Regressão Logística": LogisticRegression(max_iter=1000),
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    }
    for nome, clf in baselines.items():
        clf.fit(X_train, y_train)
        yp  = clf.predict(X_test)
        ypr = clf.predict_proba(X_test)[:,1]
        print(f"  {nome:<22} AUC={roc_auc_score(y_test,ypr):.3f}  F1={f1_score(y_test,yp):.3f}")

    nn_auc = roc_auc_score(y_test, y_proba)
    nn_f1  = f1_score(y_test, y_pred)
    print(f"  {'Rede Neural (TF)':<22} AUC={nn_auc:.3f}  F1={nn_f1:.3f} ")

    print(f"\n{sep}")
    return y_pred, y_proba, history

def avaliar_paciente(modelo, scaler, features, paciente: dict) -> str:
    """Avalia risco de um paciente individual."""
    row = pd.DataFrame([paciente])[features]
    row_scaled = scaler.transform(row)
    prob = modelo.predict(row_scaled, verbose=0)[0][0]
    nivel = " ALTO" if prob >= 0.70 else " MODERADO" if prob >= 0.40 else " BAIXO"
    return f"Probabilidade: {prob:.1%} | Risco: {nivel}"

def gerar_graficos(df, history, y_test, y_pred, y_proba):
    fig = plt.figure(figsize=(20, 12), facecolor="#0d1424")
    fig.suptitle(" Detector de Risco Cardíaco — TensorFlow", fontsize=18,
                 fontweight="bold", color="#f1f5f9", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(history.history["loss"], color=CORES[0], label="Treino", linewidth=2)
    ax1.plot(history.history["val_loss"], color=CORES[1], label="Validação", linewidth=2)
    ax1.set_title("Loss por Época", fontsize=12, fontweight="bold", color="#f1f5f9")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.legend(labelcolor="#f1f5f9", fontsize=9)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(history.history["auc"], color=CORES[2], label="Treino", linewidth=2)
    ax2.plot(history.history["val_auc"], color=CORES[3], label="Validação", linewidth=2)
    ax2.set_title("AUC por Época", fontsize=12, fontweight="bold", color="#f1f5f9")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("AUC")
    ax2.legend(labelcolor="#f1f5f9", fontsize=9)

    ax3 = fig.add_subplot(gs[0, 2])
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    ax3.plot(fpr, tpr, color=CORES[0], linewidth=2, label=f"AUC = {auc:.3f}")
    ax3.fill_between(fpr, tpr, alpha=0.1, color=CORES[0])
    ax3.plot([0,1],[0,1], "w--", linewidth=0.8, alpha=0.5)
    ax3.set_title("Curva ROC", fontsize=12, fontweight="bold", color="#f1f5f9")
    ax3.set_xlabel("FPR")
    ax3.set_ylabel("TPR")
    ax3.legend(labelcolor="#f1f5f9", fontsize=10)

    ax4 = fig.add_subplot(gs[1, 0])
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax4, cmap="Blues",
                xticklabels=["Baixo Risco","Alto Risco"],
                yticklabels=["Baixo Risco","Alto Risco"],
                annot_kws={"size": 14})
    ax4.set_title("Matriz de Confusão", fontsize=12, fontweight="bold", color="#f1f5f9")
    ax4.set_xlabel("Previsto")
    ax4.set_ylabel("Real")

    ax5 = fig.add_subplot(gs[1, 1])
    baixo = y_proba[y_test == 0]
    alto  = y_proba[y_test == 1]
    sns.kdeplot(baixo, fill=True, color=CORES[2], alpha=0.5, label="Baixo Risco", ax=ax5)
    sns.kdeplot(alto,  fill=True, color=CORES[0], alpha=0.5, label="Alto Risco",  ax=ax5)
    ax5.axvline(0.5, color="white", linestyle="--", linewidth=1.5, label="Threshold 0.5")
    ax5.set_title("Distribuição das Probabilidades", fontsize=12, fontweight="bold", color="#f1f5f9")
    ax5.set_xlabel("Probabilidade de Risco")
    ax5.legend(labelcolor="#f1f5f9", fontsize=9)

    ax6 = fig.add_subplot(gs[1, 2])
    cols_num = ["idade","pressao","colesterol","fc_maxima","depressao_st",
                "vasos_coloridos","angina","reserva_cardiaca"]
    corrs = df[cols_num + ["risco"]].corr()["risco"].drop("risco").sort_values()
    colors = [CORES[0] if v > 0 else CORES[2] for v in corrs.values]
    ax6.barh(corrs.index, corrs.values, color=colors)
    ax6.axvline(0, color="white", linewidth=0.8, alpha=0.5)
    ax6.set_title("Correlação com Risco Cardíaco", fontsize=12, fontweight="bold", color="#f1f5f9")
    ax6.set_xlabel("Correlação de Pearson")

    plt.savefig("detector_cardiaco.png", dpi=150, bbox_inches="tight", facecolor="#0d1424")
    plt.close()
    print("  detector_cardiaco.png gerado!")


def main():
    print("Gerando dataset clínico com Pandas + NumPy...")
    df = gerar_dataset(1000)
    print(f"   → Shape: {df.shape}")
    print(f"   → Taxa de risco: {df['risco'].mean():.1%}")

    eda(df)

    print("\n Iniciando pipeline TensorFlow...")
    modelo, history, X_test, y_test, X_train, y_train, scaler, features = treinar(df)

    y_pred, y_proba, history = avaliar(modelo, history, X_test, y_test, X_train, y_train)

    print("\n   PREDIÇÃO INDIVIDUAL — PACIENTES EXEMPLO")
    print(f"  {'─'*52}")
    exemplos = [
        {"nome":"Paciente A (alto risco)", "idade":65,"sexo":1,"tipo_dor":0,"pressao":160,
         "colesterol":280,"glicemia":1,"ecg":2,"fc_maxima":120,"angina":1,
         "depressao_st":2.5,"inclinacao_st":2,"vasos_coloridos":3,"talassemia":3,
         "pressao_por_idade":2.46,"reserva_cardiaca":55,"indice_risco_total":8.5},
        {"nome":"Paciente B (baixo risco)", "idade":35,"sexo":0,"tipo_dor":3,"pressao":110,
         "colesterol":180,"glicemia":0,"ecg":0,"fc_maxima":175,"angina":0,
         "depressao_st":0.0,"inclinacao_st":0,"vasos_coloridos":0,"talassemia":2,
         "pressao_por_idade":3.14,"reserva_cardiaca":140,"indice_risco_total":0.0},
    ]
    for p in exemplos:
        nome = p.pop("nome")
        resultado = avaliar_paciente(modelo, scaler, features, p)
        print(f"  {nome}: {resultado}")

    print("\n Gerando visualizações com Matplotlib + Seaborn...")
    gerar_graficos(df, history, y_test, y_pred, y_proba)

    print("\n Exportando...")
    df.to_csv("dataset_cardiaco.csv", index=False)
    modelo.save("modelo_cardiaco.keras")
    print("   → dataset_cardiaco.csv")
    print("   → modelo_cardiaco.keras")
    print("   → detector_cardiaco.png")

    print("\n Projeto concluído!")
    print("  Este modelo é para fins educacionais.")
    print("    Consulte sempre um médico para diagnósticos reais.")


if __name__ == "__main__":
    main()
