import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# Função para traduzir os valores das colunas categóricas
def traduzir_valores(df):
    df = df.copy()
    df['Gênero'] = df['Gender'].map({'Male': 'Masculino', 'Female': 'Feminino', 'Other': 'Outro'})
    df['Nível de Educação'] = df['Education_Level'].map({
        'High School': 'Ensino Médio',
        'Undergraduate': 'Graduação',
        'Postgraduate': 'Pós-Graduação'
    })
    df['Curso'] = df['Course_Name'].map({
        'Machine Learning': 'Aprend.Máquina',
        'Python Basics': 'Fund.Python',
        'Data Science': 'Ciência de Dados',
        'CyberSecurity': 'Ciberseg.',
        'Web Development' : 'Desenv.Web'
    })
    df['Estilo de Aprendizagem'] = df['Learning_Style'].map({
        'Visual': 'Visual',
        'Auditory': 'Auditivo',
        'Reading/Writing': 'Leitura/Escrita',
        'Kinesthetic': 'Cinestésico'
    })
    df['Probabilidade de Evasão'] = df['Dropout_Likelihood'].map({'Yes': 'Sim', 'No': 'Não'})
    df['Nível de Engajamento'] = df['Engagement_Level'].map({
        'Low': 'Baixo',
        'Medium': 'Médio',
        'High': 'Alto'
    })


    bins = list(range(15, 51, 5))
    labels = [f'{b} - {b+4}' for b in bins[:-1]]
    df['Idade (intervalos)'] = pd.cut(df['Age'], bins=bins, labels=labels, right=False)
    return df

# Carregar dados
df = pd.read_csv("./database/personalized_learning_dataset.csv")
df_trad = traduzir_valores(df)

st.title('🎲 Análise Exploratória dos Dados')
st.write('Esta página apresenta uma análise exploratória dos dados do dataset de aprendizado personalizado.')
genero = st.sidebar.multiselect('Gênero', options=df_trad['Gênero'].unique(), default=df_trad['Gênero'].unique())
idade = st.sidebar.multiselect('Idade (intervalos)', options=df_trad['Idade (intervalos)'].cat.categories, default=df_trad['Idade (intervalos)'].cat.categories)
nivel_educ = st.sidebar.multiselect('Nível de Educação', options=df_trad['Nível de Educação'].unique(), default=df_trad['Nível de Educação'].unique())
curso = st.sidebar.multiselect('Curso', options=df_trad['Curso'].unique(), default=df_trad['Curso'].unique())
estilo = st.sidebar.multiselect('Estilo de Aprendizagem', options=df_trad['Estilo de Aprendizagem'].unique(), default=df_trad['Estilo de Aprendizagem'].unique())
engajamento = st.sidebar.multiselect('Nível de Engajamento', options=df_trad['Nível de Engajamento'].unique(), default=df_trad['Nível de Engajamento'].unique())
feedback = st.sidebar.multiselect('Feedback (nota)', options=sorted(df_trad['Feedback_Score'].unique()), default=sorted(df_trad['Feedback_Score'].unique()))
evasao = st.sidebar.multiselect('Probabilidade de Evasão', options=df_trad['Probabilidade de Evasão'].unique(), default=df_trad['Probabilidade de Evasão'].unique())


df_filtro = df_trad[
    (df_trad['Gênero'].isin(genero)) &
    (df_trad['Idade (intervalos)'].isin(idade)) &
    (df_trad['Nível de Educação'].isin(nivel_educ)) &
    (df_trad['Curso'].isin(curso)) &
    (df_trad['Estilo de Aprendizagem'].isin(estilo)) &
    (df_trad['Probabilidade de Evasão'].isin(evasao)) &
    (df_trad['Nível de Engajamento'].isin(engajamento)) &
    (df_trad['Feedback_Score'].isin(feedback))
]

st.write(f'Dados filtrados: {len(df_filtro)} linhas')

def plot_bar(data, coluna, titulo):
    counts = data[coluna].value_counts().sort_index()
    fig, ax = plt.subplots()
    counts.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title(titulo)
    ax.set_ylabel('Quantidade')
    ax.set_xlabel(coluna)
    plt.xticks(rotation=0, ha='center')
    st.pyplot(fig)

#Gráficos de Métricas

plot_bar(df_filtro, 'Gênero', 'Distribuição por Gênero')
plot_bar(df_filtro, 'Idade (intervalos)', 'Distribuição por Idade')
plot_bar(df_filtro, 'Nível de Educação', 'Distribuição por Nível de Educação')
plot_bar(df_filtro, 'Curso', 'Distribuição por Curso')
plot_bar(df_filtro, 'Estilo de Aprendizagem', 'Distribuição por Estilo de Aprendizagem')
plot_bar(df_filtro, 'Nível de Engajamento', 'Distribuição por Nível de Engajamento')
plot_bar(df_filtro, 'Feedback_Score', 'Distribuição por Feedback')
plot_bar(df_filtro, 'Probabilidade de Evasão', 'Distribuição por Probabilidade de Evasão')

#Comparações

# Gráfico 1 - Nível de Engajamento por Nível de Educação (barras empilhadas)
st.subheader('Nível de Engajamento por Nível de Educação')
engaj_educ = df_filtro.groupby(['Nível de Educação', 'Nível de Engajamento']).size().unstack(fill_value=0)
fig, ax = plt.subplots()
engaj_educ.plot(kind='bar', stacked=True, ax=ax, color=['#1976D2','#64B5F6','#BBDEFB'])
ax.set_ylabel('Quantidade de Alunos')
ax.set_xlabel('Nível de Educação')
ax.legend(title='Nível de Engajamento')
plt.xticks(rotation=0)
st.pyplot(fig)

# Gráfico 2 - Nível de Engajamento por Idade (barras empilhadas)
st.subheader('Nível de Engajamento por Idade')
engaj_educ = df_filtro.groupby(['Idade (intervalos)', 'Nível de Engajamento']).size().unstack(fill_value=0)
fig, ax = plt.subplots()
engaj_educ.plot(kind='bar', stacked=True, ax=ax, color=['#1976D2','#64B5F6','#BBDEFB'])
ax.set_ylabel('Quantidade de Alunos')
ax.set_xlabel('Idade')
ax.legend(title='Nível de Engajamento')
plt.xticks(rotation=0)
st.pyplot(fig)

# Gráfico 3 - Probabilidade de Evasão por Nível de Engajamento (barras empilhadas em %)
st.subheader('Probabilidade de Evasão por Nível de Engajamento (proporção)')
evasao_curso = df_filtro.groupby(['Nível de Engajamento', 'Probabilidade de Evasão']).size().unstack(fill_value=0)
evasao_curso_pct = evasao_curso.div(evasao_curso.sum(axis=1), axis=0)
fig, ax = plt.subplots()
evasao_curso_pct.plot(kind='bar', stacked=True, ax=ax, color=["#2F7585","#8BC8AE"])
ax.set_ylabel('Proporção')
ax.set_xlabel('Nível de Engajamento')
ax.legend(title='Probabilidade de Evasão')
plt.xticks(rotation=0)
st.pyplot(fig)

# Gráfico 4 - Média da Nota do Exame Final por Faixa de Conclusão de Tarefas
st.subheader('Conclusão de tarefas x Nota no Exame Final')

# Criar bins para quiz e exame final (intervalos fechados 0-10, 11-20, ...)
bins_tarefa = np.arange(0, 101, 10)  # vai até 110 para pegar até 100
bins_final = np.arange(0, 101, 10)

# Criar tabela de contagem
heatmap_data, xedges, yedges = np.histogram2d(
    df_filtro['Assignment_Completion_Rate'],
    df_filtro['Final_Exam_Score'],
    bins=[bins_tarefa, bins_final]
)

fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(
    heatmap_data.T,
    cmap='Blues',
    cbar_kws={'label': 'Frequência'},
    xticklabels=[f'{int(bins_tarefa[i])}-{int(bins_tarefa[i+1]-1)}' for i in range(len(bins_tarefa)-1)],
    yticklabels=[f'{int(bins_final[i])}-{int(bins_final[i+1]-1)}' for i in range(len(bins_final)-1)],
    ax=ax
)
ax.set_xlabel('Taxa de Tarefas Concluídas (%)')
ax.set_ylabel('Nota no Exame Final (%)')

# Inverte eixo y pra crescer de baixo pra cima
ax.invert_yaxis()

st.pyplot(fig)


# Gráfico 5 - Histograma: Tentativas em Quizzes
st.subheader('Distribuição de Tentativas em Quizzes')
fig, ax = plt.subplots()
ax.hist(df_filtro['Quiz_Attempts'], bins=20, color='#2196F3', edgecolor='black')
ax.set_xlabel('Número de Tentativas')
ax.set_ylabel('Quantidade de Alunos')
st.pyplot(fig)

# Gráfico 6 - Participação em Fórum por Estilo de Aprendizagem (barra horizontal)
st.subheader('Participação em Fórum por Estilo de Aprendizagem (média de posts)')
forum_estilo = df_filtro.groupby('Estilo de Aprendizagem')['Forum_Participation'].mean().sort_values()
fig, ax = plt.subplots()
forum_estilo.plot(kind='barh', color='#1E88E5', ax=ax)
ax.set_xlabel('Média de Posts')
ax.set_ylabel('Estilo de Aprendizagem')
st.pyplot(fig)

# Gráfico 7 - Tempo Assistindo Vídeos por Tempo em Vídeos (barra horizontal)
st.subheader('Tempo Assistindo Vídeos  por Estilo de Aprendizagem (mins)')
forum_estilo = df_filtro.groupby('Estilo de Aprendizagem')['Time_Spent_on_Videos'].mean().sort_values()
fig, ax = plt.subplots()
forum_estilo.plot(kind='barh', color='#1E88E5', ax=ax)
ax.set_xlabel('Média de Tempo Assistindo Vídeos (minutos)')
ax.set_ylabel('Estilo de Aprendizagem')
st.pyplot(fig)

# Gráfico 8 - Taxa média de conclusão de tarefas por faixa etária (linha)
st.subheader('Taxa Média de Conclusão de Tarefas por Faixa Etária')
taxa_faixa = df_filtro.groupby('Idade (intervalos)')['Assignment_Completion_Rate'].mean()
fig, ax = plt.subplots()
taxa_faixa.plot(kind='line', marker='o', color='#1565C0', ax=ax)
ax.set_xlabel('Faixa Etária')
ax.set_ylabel('Taxa Média de Conclusão (%)')
st.pyplot(fig)

# Gráfico 9 - Taxa média de conclusão de tarefas por Estilo de Aprendizagem (linha)
st.subheader('Taxa Média de Conclusão de Tarefas por Estilo de Aprendizagem')
taxa_faixa = df_filtro.groupby('Estilo de Aprendizagem')['Assignment_Completion_Rate'].mean()
fig, ax = plt.subplots()
taxa_faixa.plot(kind='line', marker='o', color='#1565C0', ax=ax)
ax.set_xlabel('Estilo de Aprendizagem')
ax.set_ylabel('Taxa Média de Conclusão (%)')
st.pyplot(fig)

# Gráfico 10 - Relação entre Nota em Quiz e Nota no Exame Final (média de Final_Exam_Score para faixas de Quiz_Scores)
st.subheader('Nota em Quiz x Nota no Exame Final')

# Criar bins para quiz e exame final (intervalos fechados 0-10, 11-20, ...)
bins_quiz = np.arange(0, 101, 10)  # vai até 110 para pegar até 100
bins_final = np.arange(0, 101, 10)

# Criar tabela de contagem
heatmap_data, xedges, yedges = np.histogram2d(
    df_filtro['Quiz_Scores'],
    df_filtro['Final_Exam_Score'],
    bins=[bins_quiz, bins_final]
)

fig, ax = plt.subplots(figsize=(8,6))
sns.heatmap(
    heatmap_data.T,
    cmap='Blues',
    cbar_kws={'label': 'Frequência'},
    xticklabels=[f'{int(bins_quiz[i])}-{int(bins_quiz[i+1]-1)}' for i in range(len(bins_quiz)-1)],
    yticklabels=[f'{int(bins_final[i])}-{int(bins_final[i+1]-1)}' for i in range(len(bins_final)-1)],
    ax=ax
)
ax.set_xlabel('Nota no Quiz (%)')
ax.set_ylabel('Nota no Exame Final (%)')

# Inverte eixo y pra crescer de baixo pra cima
ax.invert_yaxis()

st.pyplot(fig)