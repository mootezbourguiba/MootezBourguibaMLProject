# --- Contenu du fichier app.py final ---
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- Configuration de la Page ---
st.set_page_config(page_title="Prédiction Prix Maison", layout="wide")

# --- Interface Utilisateur : Choix du modèle ---
st.sidebar.title("🧠 Sélection du Modèle")
model_choice = st.sidebar.selectbox("Choisissez un modèle de prédiction :", 
                                    ["Régression Linéaire", "Random Forest", "SVM"])

# --- Correspondance fichiers selon modèle ---
model_filenames = {
    "Régression Linéaire": "linear_regression_model.joblib",
    "Random Forest": "random_forest_model.joblib",
    "SVM": "svm_model.joblib"
}

scaler_filename = "standard_scaler.joblib"
columns_filename = "model_columns.joblib"

# --- Chargement du modèle, scaler et colonnes ---
try:
    model = joblib.load(model_filenames[model_choice])
    scaler = joblib.load(scaler_filename)
    model_columns = joblib.load(columns_filename)
except FileNotFoundError:
    st.error("ERREUR : Fichier(s) manquant(s). Vérifiez que tous les fichiers `.joblib` sont présents.")
    st.stop()
except Exception as e:
    st.error(f"Erreur lors du chargement : {e}")
    st.stop()

# --- Titre de l'application ---
st.title("🏠 Estimateur de Prix de Maison")
st.write(f"Modèle utilisé : **{model_choice}**")

# --- Création des colonnes d'entrée utilisateur ---
col1, col2 = st.columns(2)
input_data = {}

with col1:
    st.subheader("Caractéristiques Principales")
    input_data['area'] = st.number_input("Surface (area)", 500, 20000, 5000, step=100)
    input_data['bedrooms'] = st.number_input("Chambres (bedrooms)", 1, 10, 3)
    input_data['bathrooms'] = st.number_input("Salles de bain (bathrooms)", 1, 5, 2)
    input_data['stories'] = st.number_input("Étages (stories)", 1, 5, 2)
    input_data['parking'] = st.number_input("Parkings", 0, 5, 1)

with col2:
    st.subheader("Autres Caractéristiques")
    input_data['mainroad'] = st.radio("Proche route principale (mainroad)", ['yes', 'no'])
    input_data['guestroom'] = st.radio("Chambre d'amis (guestroom)", ['yes', 'no'])
    input_data['basement'] = st.radio("Sous-sol (basement)", ['yes', 'no'])
    input_data['hotwaterheating'] = st.radio("Chauffage eau chaude (hotwaterheating)", ['yes', 'no'])
    input_data['airconditioning'] = st.radio("Climatisation (airconditioning)", ['yes', 'no'])
    input_data['prefarea'] = st.radio("Zone préférentielle (prefarea)", ['yes', 'no'])
    furnishing = st.selectbox("État d'ameublement (furnishingstatus)", 
                              ['furnished', 'semi-furnished', 'unfurnished'], index=1)

# --- Préparation des données ---
input_df_raw = pd.DataFrame([input_data])

# Encodage binaire
bin_cols = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea']
for col in bin_cols:
    input_df_raw[col] = input_df_raw[col].map({'yes': 1, 'no': 0})

# One-hot encoding pour furnishingstatus
input_df_raw['furnishingstatus'] = furnishing
input_df_encoded = pd.get_dummies(input_df_raw, columns=['furnishingstatus'], drop_first=False)

# Alignement des colonnes avec le modèle
input_df_final = pd.DataFrame(columns=model_columns)
input_df_final = pd.concat([input_df_final, input_df_encoded], ignore_index=True).fillna(0)
input_df_final = input_df_final[model_columns]

# Option de debug
if st.checkbox("Afficher les données transformées"):
    st.write(input_df_final)

# --- Prédiction ---
if st.button("✨ Estimer le Prix !"):
    try:
        input_scaled = scaler.transform(input_df_final)
        prediction = model.predict(input_scaled)

        st.subheader("Résultat de l'Estimation")
        st.success(f"💰 Le prix estimé de la maison est : **{prediction[0]:,.0f}** (unité monétaire)")
        st.balloons()
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")

# --- Sidebar info ---
st.sidebar.info("""
**À propos :**
- Interface Streamlit multi-modèles
- Données traitées et normalisées automatiquement
- Modèles entraînés : Régression Linéaire, Random Forest, SVM
- 🔎 Projet réalisé dans le cadre d'une prédiction de prix de maisons
""")
