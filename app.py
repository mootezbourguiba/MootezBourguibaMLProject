# --- Contenu du fichier app.py final (Version Régression Linéaire uniquement) ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration de la Page ---
st.set_page_config(page_title="Prédiction Prix Maison", layout="wide")

# --- Modèle Fixe (Régression Linéaire) ---
# Puisque seul ce modèle est performant et sauvegardé, nous le définissons directement.
MODEL_NAME = "Régression Linéaire"
MODEL_FILENAME = "linear_regression_model.joblib" # Nom du fichier pour CE modèle
SCALER_FILENAME = "standard_scaler.joblib"
COLUMNS_FILENAME = "model_columns.joblib"

# --- Chargement du modèle, scaler et colonnes ---
# Simplification du chargement car on ne charge qu'un modèle spécifique
try:
    model = joblib.load(MODEL_FILENAME)
    scaler = joblib.load(SCALER_FILENAME)
    model_columns = joblib.load(COLUMNS_FILENAME)
    st.sidebar.success(f"Modèle '{MODEL_NAME}' chargé.") # Indique le succès dans la sidebar
except FileNotFoundError:
    st.error(f"ERREUR : Fichier(s) manquant(s) ({MODEL_FILENAME}, {SCALER_FILENAME}, ou {COLUMNS_FILENAME}). Vérifiez leur présence.")
    st.stop() # Arrête l'exécution si les fichiers essentiels manquent
except Exception as e:
    st.error(f"Erreur lors du chargement des fichiers : {e}")
    st.stop()

# --- Titre de l'application ---
st.title("🏠 Estimateur de Prix de Maison")
st.write(f"Utilisation du modèle : **{MODEL_NAME}**") # Affiche le nom du modèle utilisé

# --- Création des colonnes d'entrée utilisateur ---
col1, col2 = st.columns(2)
input_data = {}

with col1:
    st.subheader("Caractéristiques Principales")
    input_data['area'] = st.number_input("Surface (area)", min_value=500, max_value=20000, value=5000, step=100)
    input_data['bedrooms'] = st.number_input("Chambres (bedrooms)", min_value=1, max_value=10, value=3)
    input_data['bathrooms'] = st.number_input("Salles de bain (bathrooms)", min_value=1, max_value=5, value=2)
    input_data['stories'] = st.number_input("Étages (stories)", min_value=1, max_value=5, value=2)
    input_data['parking'] = st.number_input("Parkings", min_value=0, max_value=5, value=1)

with col2:
    st.subheader("Autres Caractéristiques")
    # Utilisation d'index pour les valeurs par défaut ('yes'=0, 'no'=1) pour plus de clarté si désiré
    input_data['mainroad'] = st.radio("Proche route principale (mainroad)", ['yes', 'no'], index=0)
    input_data['guestroom'] = st.radio("Chambre d'amis (guestroom)", ['yes', 'no'], index=1)
    input_data['basement'] = st.radio("Sous-sol (basement)", ['yes', 'no'], index=1)
    input_data['hotwaterheating'] = st.radio("Chauffage eau chaude (hotwaterheating)", ['yes', 'no'], index=1)
    input_data['airconditioning'] = st.radio("Climatisation (airconditioning)", ['yes', 'no'], index=1)
    input_data['prefarea'] = st.radio("Zone préférentielle (prefarea)", ['yes', 'no'], index=1)
    furnishing = st.selectbox("État d'ameublement (furnishingstatus)",
                              ['furnished', 'semi-furnished', 'unfurnished'], index=1) # Défaut 'semi-furnished'

# --- Préparation des données Entrées ---
# Création du DataFrame initial à partir des inputs
input_df_raw = pd.DataFrame([input_data])

# 1. Encodage binaire ('yes'/'no' -> 1/0)
bin_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in bin_cols:
    input_df_raw[col] = input_df_raw[col].map({'yes': 1, 'no': 0})

# 2. Ajout de la colonne 'furnishingstatus' avant le one-hot encoding
input_df_raw['furnishingstatus'] = furnishing

# 3. One-hot encoding pour 'furnishingstatus'
# Important: Ne pas utiliser drop_first=True ici si le modèle a été entraîné avec toutes les colonnes dummy
input_df_encoded = pd.get_dummies(input_df_raw, columns=['furnishingstatus'], drop_first=False) # Garder toutes les colonnes

# 4. Alignement et gestion des colonnes manquantes/supplémentaires
# Crée un DataFrame vide avec les colonnes attendues par le modèle
input_df_aligned = pd.DataFrame(columns=model_columns)
# Concatène avec les données encodées, en remplissant les NaN par 0 (pour les colonnes dummy manquantes)
# et en s'assurant que les colonnes non attendues sont ignorées
input_df_final = pd.concat([input_df_aligned, input_df_encoded], ignore_index=True, sort=False).fillna(0)
# Sélectionne uniquement les colonnes attendues DANS LE BON ORDRE
input_df_final = input_df_final[model_columns]


# --- Option de Debug ---
if st.checkbox("Afficher les données transformées avant scaling"):
    st.write("Données prêtes pour le scaler :")
    st.dataframe(input_df_final)

# --- Prédiction ---
if st.button("✨ Estimer le Prix !"):
    try:
        # 5. Mise à l'échelle (Scaling) avec le scaler chargé
        input_scaled = scaler.transform(input_df_final)

        # 6. Prédiction avec le modèle chargé
        prediction = model.predict(input_scaled)

        # Affichage du résultat
        st.subheader("Résultat de l'Estimation")
        # Formatage pour la lisibilité (séparateur de milliers, pas de décimales)
        st.success(f"💰 Le prix estimé de la maison est : **{prediction[0]:,.0f}** (unité monétaire)")
        st.balloons()

    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")
        st.exception(e) # Affiche plus de détails sur l'erreur pour le débogage

# --- Sidebar info (Mise à jour) ---
st.sidebar.title("ℹ️ À propos")
st.sidebar.info(f"""
**Application d'Estimation Immobilière**

- **Interface :** Streamlit
- **Modèle Utilisé :** {MODEL_NAME} (R² ≈ 0.65)
- **Prétraitement :** Les données entrées sont automatiquement encodées et normalisées.
- **Projet :** Réalisé dans le cadre d'une étude de prédiction de prix de maisons.

*Les modèles Random Forest et SVM ont été testés mais n'ont pas montré de meilleures performances sur ce jeu de données.*
""")