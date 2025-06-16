#  Original file is located at
# https://colab.research.google.com/drive/1xGwtWkZiiKSTR6sdC0Jpvc5IKo7-XhpG


from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from xgboost import XGBClassifier

df = pd.read_csv("/content/drive/MyDrive/personalized_learning_dataset.csv")

df.head()

# prompt: remova a coluna Student_ID

df = df.drop(columns=['Student_ID'], errors='ignore')
print(df.head())

# prompt: mostre os tipos de respostas unicas em 'Engagement_Level'

print(df['Engagement_Level'].unique())
print(df['Education_Level'].unique())

# prompt: realize o label encoding da coluna de Engagement_Level e Education_Level, em nível crescente ou seja, High: 2, medium: 1, low: 0. E high school: 0, undergraduate: 1 e postgraduate: 2


# Mapping for Engagement_Level
engagement_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
df['Engagement_Level_Encoded'] = df['Engagement_Level'].map(engagement_mapping)

# Mapping for Education_Level
education_mapping = {'High School': 0, 'Undergraduate': 1, 'Postgraduate': 2}
df['Education_Level_Encoded'] = df['Education_Level'].map(education_mapping)

print(df[['Engagement_Level', 'Engagement_Level_Encoded', 'Education_Level', 'Education_Level_Encoded']].head())

# prompt: faça um one-hote encoding na coluna Dropout_Likelihood onde No: 0 e Yes: 1

# Mapping for Dropout_Likelihood
dropout_mapping = {'No': 0, 'Yes': 1}
df['Dropout_Likelihood_Encoded'] = df['Dropout_Likelihood'].map(dropout_mapping)

print(df[['Dropout_Likelihood', 'Dropout_Likelihood_Encoded']].head())

# prompt: faça one-hot encoding na coluna Course_name (que é uma coluna sem ordem, apenas categorias)

# Apply one-hot encoding to 'Course_Name'
df = pd.get_dummies(df, columns=['Course_Name'], prefix='Course')

print(df.head())

"""faça um one-hot encoding na coluna Learning_Style que são tipos de aprendizados com os seguintes tipos: ['Visual' 'Reading/Writing' 'Kinesthetic' 'Auditory']. Além disso faça um one-hot encoding da coluna Gender troque 'Female' por 0 e 'Male' por 1"""

# prompt: faça um one-hot encoding na coluna Learning_Style que são tipos de aprendizados com os seguintes tipos: ['Visual' 'Reading/Writing' 'Kinesthetic' 'Auditory']. Além disso faça um one-hot encoding da coluna Gender troque 'Female' por 0 e 'Male' por 1

# One-hot encode 'Learning_Style'
df = pd.get_dummies(df, columns=['Learning_Style'], prefix='Learning_Style')

# Map 'Gender' to 0 and 1
gender_mapping = {'Female': 0, 'Male': 1}
df['Gender_Encoded'] = df['Gender'].map(gender_mapping)
df = df.drop(columns=['Gender']) # Remove the original Gender column

print(df.head())

# Estatísticas descritivas das variáveis numéricas
print("📊 Estatísticas descritivas:")
print(df.describe())

# Verificando valores únicos nas colunas categóricas (opcional, para análise futura)
print("\n🔠 Valores únicos nas colunas categóricas:")
print(df[['Education_Level', 'Engagement_Level', 'Dropout_Likelihood']].nunique())

# Verificando se há outliers (via valores máximos/mínimos comparados com média e quartis)
print("\n📈 Verificando possíveis outliers:")
for col in ['Age', 'Time_Spent_on_Videos', 'Quiz_Attempts', 'Quiz_Scores',
            'Forum_Participation', 'Assignment_Completion_Rate',
            'Final_Exam_Score', 'Feedback_Score']:
    print(f"{col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}")

df = df.dropna()

#1. Separar os dados em X (entradas) e y (saída)
X = df.drop('Dropout_Likelihood_Encoded', axis=1)
y = df['Dropout_Likelihood_Encoded']

X = X.drop(['Education_Level', 'Engagement_Level', 'Dropout_Likelihood'], axis=1)
print(X.select_dtypes(include='object').columns)

# prompt: troque as colunas Course_Cybersecurity                 bool
# Course_Data Science                  bool
# Course_Machine Learning              bool
# Course_Python Basics                 bool
# Course_Web Development               bool
# Learning_Style_Auditory              bool
# Learning_Style_Kinesthetic           bool
# Learning_Style_Reading/Writing       bool
# Learning_Style_Visual                bool
# onde true = 1 e false = 0

# Converta as colunas booleanas para inteiros
bool_cols_to_convert = [
    'Course_Cybersecurity',
    'Course_Data Science',
    'Course_Machine Learning',
    'Course_Python Basics',
    'Course_Web Development',
    'Learning_Style_Auditory',
    'Learning_Style_Kinesthetic',
    'Learning_Style_Reading/Writing',
    'Learning_Style_Visual'
]

for col in bool_cols_to_convert:
  if col in df.columns:
    df[col] = df[col].astype(int)

print(df.head())
df[bool_cols_to_convert].dtypes


# 1. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Calcular scale_pos_weight
neg = sum(y_train == 0)
pos = sum(y_train == 1)
scale_pos_weight = 3


# 2. Pipeline: SMOTE + XGBoost
pipeline = Pipeline([
    ('xgb', XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight,
        n_estimators=200,
        learning_rate=0.05
    ))
])

# 3. Grid de parâmetros (prefixar com 'xgb__')
param_grid = {
    'xgb__n_estimators': [100, 200],
    'xgb__max_depth': [3, 6],
    'xgb__learning_rate': [0.01, 0.1],
    'xgb__subsample': [0.8, 1.0]
}

# 4. Cross-validation com scoring focado em recall
grid = GridSearchCV(
    pipeline, param_grid,
    scoring='recall',
    cv=StratifiedKFold(n_splits=5),
    n_jobs=1,
    verbose=1
)

# 5. Treinar o modelo
grid.fit(X_train, y_train)

# 6. Avaliar no conjunto de teste
y_proba = grid.predict_proba(X_test)[:, 1]
threshold = 0.5  # Experimente 0.3, 0.25, 0.2...
y_pred = (y_proba >= threshold).astype(int)

print("\nMelhores parâmetros encontrados:", grid.best_params_)

print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred, digits=3))

# 7. Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sem Risco", "Com Risco"])
disp.plot(cmap="Reds")
plt.show()


# 1. Pipeline: SMOTE + Regressão Logística
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('logreg', LogisticRegression(max_iter=1000, random_state=42))
])

# 2. Grid de parâmetros
param_grid = {
    'logreg__C': [0.01, 0.1, 1, 10],  # regularização
    'logreg__penalty': ['l1', 'l2']
}

# 3. Cross-validation
grid = GridSearchCV(
    pipeline, param_grid,
    scoring='recall',
    cv=StratifiedKFold(n_splits=5),
    n_jobs=1,
    verbose=1
)

# 4. Treinar o modelo
grid.fit(X_train, y_train)


# 5. Avaliar no conjunto de teste
y_pred = grid.predict(X_test)

print("\nMelhores parâmetros encontrados:", grid.best_params_)

print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred, digits=3))

# 6. Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sem Risco", "Com Risco"])
disp.plot(cmap="Reds")
plt.show()


# 1. Obter probabilidades da classe "Com Risco" (classe 1)
y_proba = grid.predict_proba(X_test)[:, 1]

# 2. Ajustar o threshold (pode testar valores como 0.4, 0.3, 0.2...)
threshold = 0.5
y_pred_thresh = (y_proba >= threshold).astype(int)

# 3. Exibir novo relatório de classificação
print(f'Classification Report (Threshold = {threshold}):')
print(classification_report(y_test, y_pred_thresh, target_names=['Sem Risco', 'Com Risco']))

# 4. Matriz de confusão
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_thresh, display_labels=['Sem Risco', 'Com Risco'], cmap='Reds')
plt.show()


# 1. Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Pipeline: SMOTE + Random Forest
pipeline = Pipeline([
    ('smote', SMOTE(random_state=42)),
    ('rf', RandomForestClassifier(
        class_weight='balanced',
        random_state=42
    ))
])

# 3. Grid de parâmetros (prefixar com 'rf__')
param_grid = {
    'rf__n_estimators': [100, 200],
    'rf__max_depth': [None, 10, 20],
    'rf__min_samples_split': [2, 5]
}

# 4. Cross-validation com scoring focado em recall
grid = GridSearchCV(
    pipeline, param_grid,
    scoring='recall',
    cv=StratifiedKFold(n_splits=5),
    n_jobs=-1,
    verbose=1
)

# 5. Treinar o modelo
grid.fit(X_train, y_train)

# 6. Previsões com threshold padrão
best_model = grid.best_estimator_
y_probs = best_model.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= 0.305).astype(int)

# 7. Avaliar
print("\nMelhores parâmetros encontrados:", grid.best_params_)
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred, digits=3))

# 8. Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Sem Risco", "Com Risco"])
disp.plot(cmap="Reds")
plt.title("Matriz de Confusão - Random Forest")
plt.show()


# 1. Verificar a distribuição das classes
print("Distribuição das classes (y):")
print(y.value_counts(normalize=True))  # Proporções
print(y.value_counts())                # Contagem absoluta

# 2. Verificar shape do dataset
print("\nShape do dataset:")
print(X.shape)

# 3. Verificar tipos de dados
print("\nTipos de dados:")
print(X.dtypes)

# 4. Importância das features (com nomes reais)
best_model = grid.best_estimator_.named_steps['xgb']
feat_importances = pd.Series(best_model.feature_importances_, index=X.columns)

# 5. Plotar as 10 features mais importantes
plt.figure(figsize=(10, 6))
feat_importances.nlargest(20).plot(kind='barh')
plt.title("Importância das Features (nomes reais)")
plt.xlabel("Importância")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# 6. Matriz de correlação das features numéricas
plt.figure(figsize=(12, 8))
sns.heatmap(X.corr(numeric_only=True), cmap='coolwarm', center=0, annot=False)
plt.title("Matriz de Correlação (features numéricas)")
plt.tight_layout()
plt.show()