import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Leitura do dataset
file_path = '/kaggle/input/personalized-learning-and-adaptive-education-dataset/personalized_learning_dataset.csv'
df = pd.read_csv(file_path)

# Traduções agrupadas
df = df.replace({
    'Engagement_Level': {'Low': 'Baixo', 'Medium': 'Médio', 'High': 'Alto'},
    'Gender': {'Male': 'Masculino', 'Female': 'Feminino', 'Other': 'Outro'},
    'Learning_Style': {
        'Visual': 'Visual',
        'Auditory': 'Auditivo',
        'Kinesthetic': 'Cinestésico',
        'Reading/Writing': 'Leitura/Escrita'
    },
    'Education_Level': {
        'High School': 'Ensino Médio',
        'Technical': 'Técnico',
        'Undergraduate': 'Graduação',
        'Postgraduate': 'Pós-graduação'
    }
})

# Coluna extra para nome amigável do estilo de aprendizagem (pode ser usada em gráficos)
df['Estilo_PT'] = df['Learning_Style']
df['Engagement_Level_PT'] = df['Engagement_Level']

# Gráfico: Nível de Engajamento
engagement_counts = df["Engagement_Level_PT"].value_counts().reindex(["Baixo", "Médio", "Alto"])

sns.set(style="whitegrid")
plt.figure(figsize=(8, 5))
sns.barplot(x=engagement_counts.index, y=engagement_counts.values, palette="Blues")

plt.title("Distribuição do Nível de Engajamento", fontsize=14)
plt.xlabel("Nível de Engajamento")
plt.ylabel("Número de Estudantes")
plt.tight_layout()
plt.savefig("grafico_nivel_engajamento.png", dpi=300)
plt.show()

# Distribuição das notas dos quizzes por curso
plt.figure(figsize=(12, 6))
sns.violinplot(x='Course_Name', y='Quiz_Scores', data=df)
plt.xticks(rotation=45)
plt.title('Distribuição das Pontuações no Quiz por Curso')
plt.tight_layout()
plt.savefig("violin_quiz_scores_por_curso.png")
plt.show()

# Distribuição das notas das provas finais por curso
plt.figure(figsize=(12, 6))
sns.violinplot(x='Course_Name', y='Final_Exam_Score', data=df)
plt.xticks(rotation=45)
plt.title('Distribuição das Pontuações Finais por Curso')
plt.tight_layout()
plt.savefig("violin_final_scores_por_curso.png")
plt.show()

# Perfil demográfico
variaveis = ['Gender', 'Age', 'Learning_Style', 'Education_Level']
titulos = ['Gênero', 'Idade', 'Estilo de Aprendizado', 'Nível de Escolaridade']

fig, axs = plt.subplots(2, 2, figsize=(14, 10))
axs = axs.flatten()

for i, var in enumerate(variaveis):
    df[var].value_counts().plot(kind='bar', ax=axs[i], color='skyblue', edgecolor='black')
    axs[i].set_title(titulos[i])
    axs[i].set_ylabel('Número de participantes')
    axs[i].set_xlabel('')
    axs[i].grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig("perfil_demografico.png", dpi=300)
plt.show()

# Estilo de aprendizagem vs tempo em vídeos
plt.figure(figsize=(10, 6))
sns.boxplot(x='Estilo_PT', y='Time_Spent_on_Videos', data=df, palette='Set2')
plt.title('Tempo Assistindo Vídeos por Estilo de Aprendizagem', fontsize=14)
plt.xlabel('Estilo de Aprendizagem')
plt.ylabel('Tempo em Vídeos (minutos)')
plt.tight_layout()
plt.savefig('estilo_aprendizagem_vs_tempo_videos.png', dpi=300)
plt.show()

# Estilo de aprendizagem vs participação em fóruns
plt.figure(figsize=(10, 6))
sns.boxplot(x='Estilo_PT', y='Forum_Participation', data=df, palette='Set2')
plt.title('Participação em Fóruns por Estilo de Aprendizagem', fontsize=14)
plt.xlabel('Estilo de Aprendizagem')
plt.ylabel('Número de Postagens no Fórum')
plt.tight_layout()
plt.savefig('estilo_aprendizagem_vs_participacao_forum.png', dpi=300)
plt.show()