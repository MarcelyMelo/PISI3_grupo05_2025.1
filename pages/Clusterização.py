import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
from math import pi
import numpy as np

st.title("🔍 Clusterização de Estudantes")

st.markdown("""
Essa análise utiliza a técnica de clusterização para identificar **grupos de estudantes semelhantes** com base em seu comportamento e desempenho. Isso pode auxiliar na personalização de estratégias educacionais.
""")

# Carregar dados
df = pd.read_csv("./database/personalized_learning_dataset.csv")

# Traduzir valores categóricos
df['Curso'] = df['Course_Name'].map({
    'Machine Learning': 'Aprendizado de Máquina',
    'Python Basics': 'Fundamentos de Python',
    'Data Science': 'Ciência de Dados',
    'CyberSecurity': 'Cibersegurança',
    'Artificial Intelligence': 'Inteligência Artificial'
})
df['Nível de Engajamento'] = df['Engagement_Level'].map({
    'Low': 'Baixo',
    'Medium': 'Médio',
    'High': 'Alto'
})
df['Gênero'] = df['Gender'].map({
    'Male': 'Masculino',
    'Female': 'Feminino',
    'Other': 'Outro'
})
df['Nível de Educação'] = df['Education_Level'].map({
    'High School': 'Ensino Médio',
    'Undergraduate': 'Graduação',
    'Postgraduate': 'Pós-Graduação'
})
df['Estilo de Aprendizagem'] = df['Learning_Style'].map({
    'Visual': 'Visual',
    'Auditory': 'Auditivo',
    'Reading/Writing': 'Leitura/Escrita',
    'Kinesthetic': 'Cinestésico'
})



# Renomear colunas para português
df.rename(columns={
    'Age': 'Idade',
    'Time_Spent_on_Videos': 'Tempo em Vídeos (min)',
    'Quiz_Attempts': 'Tentativas de Quiz',
    'Quiz_Scores': 'Nota no Quiz (%)',
    'Forum_Participation': 'Participação no Fórum',
    'Assignment_Completion_Rate': 'Conclusão de Tarefas (%)',
    'Final_Exam_Score': 'Nota Final (%)',
    'Feedback_Score': 'Nota de Feedback',
    'Dropout_Likelihood': 'Probab de Evasão',
}, inplace=True)

# Adicionar ambas as versões da coluna de evasão
df['Probabilidade de Evasão'] = df['Probab de Evasão'].map({'Yes': 1, 'No': 0})  # Para clusterização
# Selecionar colunas numéricas para clusterização
colunas_cluster = [
    'Tempo em Vídeos (min)',
    'Tentativas de Quiz',
    'Nota no Quiz (%)',
    'Participação no Fórum',
    'Conclusão de Tarefas (%)',
    'Nota Final (%)',
    'Nota de Feedback',
    'Probabilidade de Evasão',
    'Idade'
]

df_cluster = df[colunas_cluster].dropna()

# Padronização
scaler = StandardScaler()
dados_padronizados = scaler.fit_transform(df_cluster)

# Sidebar: Parâmetros
st.sidebar.header("Parâmetros de Clusterização")
n_clusters = st.sidebar.slider("Número de Clusters", 2, 10, 3)

# KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(dados_padronizados)
df_cluster['Cluster'] = clusters

# PCA para visualização
pca = PCA(n_components=2)
pca_result = pca.fit_transform(dados_padronizados)
df_cluster['Componente 1'] = pca_result[:, 0]
df_cluster['Componente 2'] = pca_result[:, 1]

# Gráfico de dispersão dos clusters
st.subheader("Visualização dos Clusters (PCA)")
fig1, ax1 = plt.subplots()
for c in range(n_clusters):
    grupo = df_cluster[df_cluster['Cluster'] == c]
    ax1.scatter(grupo['Componente 1'], grupo['Componente 2'], label=f'Cluster {c}')
ax1.set_xlabel("Componente 1")
ax1.set_ylabel("Componente 2")
ax1.set_title("Projeção PCA dos Clusters")
ax1.legend()
st.pyplot(fig1)

# Tabela com médias por cluster
st.subheader("📊 Médias por Cluster")
st.dataframe(df_cluster.groupby("Cluster").mean().round(2))

# Gráficos de barra com médias por cluster
def plot_bar_cluster(coluna, titulo, cor):
    st.subheader(titulo)
    fig, ax = plt.subplots()
    df_cluster.groupby("Cluster")[coluna].mean().plot(kind='bar', color=cor, ax=ax)
    ax.set_ylabel("Média")
    ax.set_xlabel("Cluster")
    ax.set_title(titulo)
    plt.xticks(rotation=0)
    st.pyplot(fig)

plot_bar_cluster('Nota Final (%)', '🎓 Nota Média no Exame Final por Cluster', "#4FC3F7")
plot_bar_cluster('Idade', '🧍‍♂️Idade Média por Cluster', "#FFB85B")
plot_bar_cluster('Conclusão de Tarefas (%)', '📈 Taxa Média de Conclusão de Tarefas por Cluster', "#81C784")
plot_bar_cluster('Participação no Fórum', '💬 Participação Média no Fórum por Cluster', "#BA68C8")
plot_bar_cluster('Nota no Quiz (%)', '📚 Nota Média nos Quizzes por Cluster', "#FFD54F")
plot_bar_cluster('Nota de Feedback', '⭐ Nota Média de Feedback por Cluster', "#FF8A65")

# Gráficos adicionais de métricas numéricas por cluster
plot_bar_cluster('Tempo em Vídeos (min)', '🕒 Tempo Médio em Vídeos por Cluster', "#90CAF9")
plot_bar_cluster('Tentativas de Quiz', '🔁 Média de Tentativas de Quiz por Cluster', "#F48FB1")


# 📉 Gráfico: Evasão por Cluster
df['Evasão'] = df['Probab de Evasão'].map({'Yes': 'Sim', 'No': 'Não'})  # Para gráfico
df_cluster.loc[:, 'Evasão'] = df.loc[df_cluster.index, 'Evasão']

st.subheader("📉 Proporção de Evasão por Cluster")

evasao_cluster = df_cluster.groupby('Cluster')['Evasão'].value_counts(normalize=True).unstack().fillna(0) * 100

fig3, ax3 = plt.subplots()
evasao_cluster.plot(kind='bar', stacked=True, ax=ax3, color=["#64b5f6", "#ef5350"])
ax3.set_ylabel("Percentual (%)")
ax3.set_title("Proporção de Evasão por Cluster")
ax3.legend(title="Probabilidade de Evasão")
plt.xticks(rotation=0)
st.pyplot(fig3)

# Gráfico de Radar: comparação de todas as métricas por cluster
st.subheader("📌 Comparação Geral de Métricas por Cluster (Radar)")

# Preparar dados
df_radar = df_cluster.groupby("Cluster")[colunas_cluster].mean().reset_index()
df_radar = df_radar.set_index('Cluster')

# Normalizar os dados entre 0 e 1
df_radar_norm = (df_radar - df_radar.min()) / (df_radar.max() - df_radar.min())

# Plot
fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
categorias = df_radar_norm.columns
N = len(categorias)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

for i, row in df_radar_norm.iterrows():
    valores = row.tolist()
    valores += valores[:1]  # fechar o gráfico
    ax.plot(angles, valores, label=f'Cluster {i}')
    ax.fill(angles, valores, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categorias, fontsize=8)
ax.set_title("Radar de Métricas por Cluster", y=1.08)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
st.pyplot(fig)

# Heatmap com comparação entre clusters
st.subheader("🔥 Heatmap de Métricas por Cluster")

fig2, ax2 = plt.subplots(figsize=(10, 4))
sns.heatmap(df_radar.T, cmap='Blues', annot=True, fmt=".1f", ax=ax2)
ax2.set_title("Médias das Métricas por Cluster")
st.pyplot(fig2)
