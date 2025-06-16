import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from xgboost import XGBClassifier

st.title("🎓 Classificação de Evasão de Estudantes (Com Balanceamento SMOTE)")

st.markdown("""
Este aplicativo usa diferentes modelos de classificação para prever a **probabilidade de evasão (dropout)** de estudantes.  
**Inclui balanceamento com SMOTE e novos modelos!**
""")

# Carregar dataset
df = pd.read_csv("./database/personalized_learning_dataset.csv")

# Pré-processamento
st.header("🔍 Pré-processamento dos Dados")

df = df.drop(['Student_ID'], axis=1)

# Label Encoding do target
le = LabelEncoder()
df['Dropout_Likelihood_Encoded'] = le.fit_transform(df['Dropout_Likelihood'])

# One-Hot Encoding
X = df.drop(['Dropout_Likelihood', 'Dropout_Likelihood_Encoded'], axis=1)
X = pd.get_dummies(X, drop_first=True)
y = df['Dropout_Likelihood_Encoded']

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

st.write("📏 Dimensão dos dados após pré-processamento:", X.shape)

# Split antes do SMOTE
test_size = st.sidebar.slider("📏 Porcentagem para Teste", 0.1, 0.5, 0.3)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42, stratify=y)

# SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

st.write("📈 Dimensão do treino após SMOTE:", X_train_res.shape)

# Escolha do modelo
st.sidebar.header("⚙️ Escolha do Modelo")
model_choice = st.sidebar.selectbox(
    "Modelo:",
    ("Regressão Logística", "Random Forest", "Balanced Random Forest", "SVM", "XGBoost")
)

# Threshold customizado
threshold = st.sidebar.slider("Threshold de Decisão", 0.1, 0.9, 0.5, 0.05)

if st.sidebar.button("🔎 Treinar Modelo"):
    # Instanciar modelo
    if model_choice == "Regressão Logística":
        model = LogisticRegression(max_iter=1000, random_state=42)
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(class_weight='balanced', random_state=42)
    elif model_choice == "Balanced Random Forest":
        model = BalancedRandomForestClassifier(random_state=42)
    elif model_choice == "SVM":
        model = SVC(probability=True, random_state=42)
    else:  # XGBoost
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight, random_state=42)

    # Treinar
    model.fit(X_train_res, y_train_res)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Predição com threshold customizado
    if y_proba is not None:
        y_pred = (y_proba >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)

    st.success(f"✅ Modelo **{model_choice}** treinado!")

    # Matriz de Confusão
    st.subheader("📊 Matriz de Confusão")
    fig_cf, ax_cf = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Não Evadiu", "Evadiu"]).plot(cmap='Blues', ax=ax_cf)
    ax_cf.set_title("Matriz de Confusão")
    st.pyplot(fig_cf)

    # Relatório de Classificação
    st.subheader("📄 Relatório de Classificação")
    report = classification_report(y_test, y_pred, target_names=["Não Evadiu", "Evadiu"], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.dataframe(report_df.style.format(precision=2))

    # AUC
    if y_proba is not None:
        auc_score = roc_auc_score(y_test, y_proba)
        st.info(f"Área sob a Curva ROC (AUC): **{auc_score:.2f}**")

    # Curva ROC
    if y_proba is not None:
        st.subheader("📈 Curva ROC")
        fig_roc, ax_roc = plt.subplots()
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax_roc, name=model_choice)
        ax_roc.set_title("Curva ROC")
        st.pyplot(fig_roc)

    # Importância das Features
    if hasattr(model, "feature_importances_"):
        st.subheader("🌳 Importância das Features")
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        fig_imp, ax_imp = plt.subplots(figsize=(8, 5))
        sns.barplot(x=importances, y=importances.index, palette="viridis", ax=ax_imp)
        ax_imp.set_title("Importância das Variáveis")
        st.pyplot(fig_imp)

    elif model_choice == "Regressão Logística":
        st.subheader("⚖️ Coeficientes das Variáveis")
        coef_df = pd.DataFrame({
            'Variável': X.columns,
            'Coeficiente': model.coef_[0]
        }).sort_values(by="Coeficiente", key=abs, ascending=False)
        fig_coef, ax_coef = plt.subplots(figsize=(8, 5))
        sns.barplot(x="Coeficiente", y="Variável", data=coef_df, palette="coolwarm", ax=ax_coef)
        ax_coef.set_title("Coeficiente das Variáveis")
        st.pyplot(fig_coef)

    # Input manual para predição
    st.subheader("🧑‍💻 Previsão com Dados Inseridos")
    user_input = []
    st.markdown("Preencha os valores para cada variável:")

    for col in X.columns:
        value = st.number_input(f"{col}:", value=0.0)
        user_input.append(value)

    if st.button("Prever com Dados Inseridos"):
        user_data = scaler.transform([user_input])
        pred_user = model.predict(user_data)[0]
        proba_user = model.predict_proba(user_data)[0][1] if hasattr(model, "predict_proba") else None
        resultado = "Evadiu" if pred_user == 1 else "Não Evadiu"
        st.success(f"Previsão: **{resultado}**")
        if proba_user is not None:
            st.info(f"Probabilidade de Evasão: **{proba_user:.2%}**")