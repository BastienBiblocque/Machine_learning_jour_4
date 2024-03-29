import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
import re

# Charger le CSV
chemin_fichier = "./data_sets/train.csv"

df = pd.read_csv(chemin_fichier)

# Ajouter une colonne avec le nombre de caractères pour chaque texte
df['Nombre_de_caracteres'] = df['Text'].apply(len)
df['Nombre_de_mots'] = df['Text'].apply(lambda x: len(x.split()))

df['Text'] = df['Text'].str.lower()
df['Text_Tag'] = df['Text_Tag'].str.lower()

# Enlever les caractères spéciaux et non alphabétiques de la colonne "Text"
df['Text'] = df['Text'].apply(lambda x: re.sub(r'[^a-z\s]', '', str(x)))

# Enlever les caractères spéciaux et non alphabétiques de la colonne "Text_Tag"
df['Text_Tag'] = df['Text_Tag'].apply(lambda x: re.sub(r'[^a-z\s]', '', str(x)))

df['Text'] = df['Text'].apply(lambda x: ' '.join([mot for mot in x.split() if len(mot) > 3]))

X = df[['Nombre_de_caracteres', 'Nombre_de_mots']]
y = df['Labels']

# Instancier le RandomUnderSampler
rus = RandomUnderSampler(random_state=42)

# Appliquer le sous-échantillonnage
X_resampled, y_resampled = rus.fit_resample(X, y)

# Créer un nouveau DataFrame avec les données équilibrées
df_resampled = pd.DataFrame(X_resampled, columns=['Nombre_de_caracteres', 'Nombre_de_mots'])
df_resampled['Labels'] = y_resampled.astype(str)

# Séparer les caractéristiques (X) et la cible (y)
X_resampled = df_resampled[['Nombre_de_caracteres', 'Nombre_de_mots']]
y_resampled = df_resampled['Labels']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Créer un modèle d'arbre de décision
model = DecisionTreeClassifier(random_state=42)

# Entraîner le modèle sur l'ensemble d'entraînement
model.fit(X_train, y_train)

# Prédire sur l'ensemble de test
y_pred = model.predict(X_test)


def diplay_precision():
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Précision du modèle : {accuracy * 100:.2f}%")
#     Précision du modèle : 17.78%


def display_rapport_classification():
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))
# Rapport de classification :
#               precision    recall  f1-score   support
#
#            0       0.20      0.24      0.22       189
#            1       0.17      0.24      0.20       165
#            2       0.14      0.17      0.15       156
#            3       0.20      0.17      0.19       158
#            4       0.15      0.10      0.12       178
#            5       0.20      0.14      0.17       161
#
#     accuracy                           0.18      1007
#    macro avg       0.18      0.18      0.17      1007
# weighted avg       0.18      0.18      0.17      1007


def display_matrice_confusion():
    # Afficher la matrice de confusion
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("\nMatrice de confusion :")
    print(conf_matrix)
    # Matrice de confusion :
    # [[45 40 37 24 22 21]
    #  [37 40 33 24 15 16]
    #  [35 34 26 18 23 20]
    #  [39 36 20 27 19 17]
    #  [37 44 36 25 18 18]
    #  [30 44 28 14 22 23]]


def display_decision_tree():
    plt.figure(figsize=(12, 8))
    tree.plot_tree(model, feature_names=['Nombre_de_caracteres', 'Nombre_de_mots'],
                   class_names=df_resampled['Labels'].unique(), filled=True, rounded=True)
    plt.show()


def prediction_fichier_test():
    chemin_fichier_test = "./data_sets/test.csv"
    df_test = pd.read_csv(chemin_fichier_test)

    df_test['Nombre_de_caracteres'] = df_test['Text'].apply(len)
    df_test['Nombre_de_mots'] = df_test['Text'].apply(lambda x: len(x.split()))
    df_test['Text'] = df_test['Text'].str.lower()
    df_test['Text_Tag'] = df_test['Text_Tag'].str.lower()

    df_test['Text'] = df_test['Text'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))
    df_test['Text_Tag'] = df_test['Text_Tag'].apply(lambda x: re.sub(r'[^a-z\s]', '', x))
    df_test['Text'] = df['Text'].apply(lambda x: ' '.join([mot for mot in x.split() if len(mot) > 3]))

    X_test = df_test[['Nombre_de_caracteres', 'Nombre_de_mots']]
    y_pred_test = model.predict(X_test)

    print("Prédictions sur le fichier de test :")
    print(y_pred_test)
    # ['4' '1' '1'... '1' '3' '0']


diplay_precision()
display_rapport_classification()
display_matrice_confusion()
# display_decision_tree()
prediction_fichier_test()
