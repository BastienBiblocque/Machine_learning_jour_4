import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import re
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Charger le CSV
chemin_fichier = "./data_sets/train.csv"
df = pd.read_csv(chemin_fichier)

# Prétraitement des données
df['Text'] = df['Text'].str.lower()
df['Text'] = df['Text'].apply(lambda x: re.sub(r'[^a-z\s]', '', str(x)))
df['Text'] = df['Text'].apply(lambda x: ' '.join([mot for mot in x.split() if len(mot) > 2]))

# Séparation des données
X = df['Text']
y = df['Labels']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorisation des données textuelles
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Convertir les données en tensors PyTorch
X_train_tensor = torch.tensor(X_train_vec.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.int64)
X_test_tensor = torch.tensor(X_test_vec.toarray(), dtype=torch.float32)

# Définir le modèle PyTorch
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Initialiser le modèle
input_size = X_train_vec.shape[1]
hidden_size = 100
output_size = 6
model = SimpleNN(input_size, hidden_size, output_size)

# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Entraîner le modèle
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

# Prédictions sur l'ensemble de test
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, y_pred_tensor = torch.max(outputs, 1)
    y_pred = y_pred_tensor.numpy()

# Mesure de la précision
accuracy = accuracy_score(y_test, y_pred)
print(f"Précision du modèle : {accuracy * 100:.2f}%")


# Prédictions sur l'ensemble d'entraînement
with torch.no_grad():
    outputs_train = model(X_train_tensor)
    _, y_pred_train_tensor = torch.max(outputs_train, 1)
    y_pred_train = y_pred_train_tensor.numpy()

# Créer la matrice de confusion pour l'ensemble d'entraînement
conf_matrix_train = confusion_matrix(y_train, y_pred_train)

# Afficher la matrice de confusion pour l'ensemble d'entraînement
disp_train = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_train, display_labels=df['Labels'].unique())
disp_train.plot(cmap='viridis', values_format='d')
plt.show()


# Charger le fichier de test (remplace 'chemin_fichier_test' par le chemin vers ton fichier de test)
chemin_fichier_test = "./data_sets/test.csv"
df_test = pd.read_csv(chemin_fichier_test)

# Prétraitement des données sur le fichier de test
df_test['Text'] = df_test['Text'].str.lower()
df_test['Text'] = df_test['Text'].apply(lambda x: re.sub(r'[^a-z\s]', '', str(x)))
df_test['Text'] = df_test['Text'].apply(lambda x: ' '.join([mot for mot in x.split() if len(mot) > 2]))

# Vectorisation des données textuelles
X_test_vec = vectorizer.transform(df_test['Text'])

# Convertir les données en tensors PyTorch
X_test_tensor = torch.tensor(X_test_vec.toarray(), dtype=torch.float32)

# Prédictions sur le fichier de test
with torch.no_grad():
    outputs = model(X_test_tensor)
    _, y_pred_tensor = torch.max(outputs, 1)
    y_pred = y_pred_tensor.numpy()

# Afficher les prédictions
df_test['Predictions'] = y_pred
print(df_test[['Text', 'Predictions']])
