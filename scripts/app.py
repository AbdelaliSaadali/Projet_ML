import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pickle

# Page Config
st.set_page_config(page_title="ML Project Portfolio", layout="wide")

# Sidebar Navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Company Classification", "Salary Prediction (Marouane)", "Salary Prediction (Asaad)"])

# ==========================================
# 1. Company Classification App
# ==========================================
def show_company_classification():
    st.title("🏢 Company Classification AI")
    st.markdown("""
    This application uses Machine Learning (K-Means Clustering) to classify companies into 4 categories based on their monthly worker data:
    - **Entreprises stables**: Consistent work days, high full-time ratio.
    - **Entreprises saisonnières**: Fewer work days, lower pay (likely part-time/seasonal).
    - **Entreprises irrégulières**: High variance in work days, mixed workforce.
    - **Entreprises potentiellement frauduleuses**: Anomalous data (e.g., extremely high salaries).
    """)

    # Load Data
    @st.cache_data
    def load_data():
        if os.path.exists('processed_features.csv'):
            df = pd.read_csv('processed_features.csv')
            return df
        return None

    df = load_data()

    if df is None:
        st.error("Data not found. Please run the analysis notebook/script first.")
        return

    # Load Adherents for Names
    try:
        adherents_path = '../Data CNSS/ADHERENTS.csv'
        if os.path.exists(adherents_path):
            adherents = pd.read_csv(adherents_path)
            adherents = adherents.rename(columns={'bank_adherent_adherentMandataire': 'ID_adherent'})
            adherents = adherents[['ID_adherent', 'companyName']]
            adherents = adherents.drop_duplicates(subset=['ID_adherent'])
            df = df.merge(adherents, on='ID_adherent', how='left')
            df['companyName'] = df['companyName'].fillna("Unknown")
        else:
            df['companyName'] = "Unknown"
            
    except Exception as e:
        st.warning(f"Could not load company names: {e}")
        df['companyName'] = "Unknown"

    # Sort by ID_adherent
    df = df.sort_values('ID_adherent')

    # Cluster Mapping
    CLUSTER_MAP = {
        0: "Entreprises saisonnières",
        1: "Entreprises irrégulières",
        2: "Entreprises potentiellement frauduleuses",
        3: "Entreprises stables"
    }

    COLOR_MAP = {
        "Entreprises stables": "green",
        "Entreprises saisonnières": "orange",
        "Entreprises irrégulières": "yellow",
        "Entreprises potentiellement frauduleuses": "red"
    }

    if 'cluster' in df.columns:
        df['Category'] = df['cluster'].map(CLUSTER_MAP)
    else:
        st.error("Cluster column not found in data.")
        return

    # Sidebar - Company Selection
    st.sidebar.header("Select Company")
    df['label'] = df['ID_adherent'].astype(str) + " - " + df['companyName']
    selected_label = st.sidebar.selectbox("Choose a Company", df['label'].unique())
    selected_id = int(selected_label.split(" - ")[0])

    # Main Content
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Company Details")
        company_data = df[df['ID_adherent'] == selected_id].iloc[0]
        
        category = company_data['Category']
        color = COLOR_MAP.get(category, "grey")
        
        st.markdown(f"### Classification: :{color}[{category}]")
        
        st.metric("Number of Workers", int(company_data['num_workers']))
        st.metric("Average Days Worked", f"{company_data['avg_days']:.2f}")
        st.metric("Average Salary", f"{company_data['avg_salary']:.2f} MAD")
        st.metric("Full Time Ratio", f"{company_data['full_time_ratio']*100:.1f}%")
        
        st.markdown("---")
        st.write("**Raw Stats:**")
        st.write(company_data[['std_days', 'std_salary', 'total_salary']])

    with col2:
        st.subheader("Cluster Visualization")
        
        # Scatter Plot 1
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='avg_days', y='std_days', hue='Category', palette=COLOR_MAP, alpha=0.6, ax=ax)
        sns.scatterplot(x=[company_data['avg_days']], y=[company_data['std_days']], color='black', s=200, marker='X', label=f"Selected ({selected_id})", ax=ax)
        plt.title("Work Stability: Average Days vs Standard Deviation")
        st.pyplot(fig)
        
        # Scatter Plot 2
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.scatterplot(data=df, x='avg_days', y='avg_salary', hue='Category', palette=COLOR_MAP, alpha=0.6, ax=ax2)
        sns.scatterplot(x=[company_data['avg_days']], y=[company_data['avg_salary']], color='black', s=200, marker='X', label=f"Selected ({selected_id})", ax=ax2)
        plt.yscale('log')
        plt.title("Salary vs Work Days (Log Scale Salary)")
        st.pyplot(fig2)

    st.subheader("All Companies Data")
    st.dataframe(df)

# ==========================================
# 2. Salary Prediction (Marouane)
# ==========================================
def show_salary_prediction_marouane():
    st.title("🇲🇦 Prédiction de Salaire IT - Marché Marocain 2025 (Marouane)")
    
    model_path = '../other MLs/ML-Project-marouane/model_salaire_IT_MAROC_2025.pkl'
    
    @st.cache_resource
    def load_marouane_model():
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            st.error(f"Model file not found at {model_path}")
            return None

    model_params = load_marouane_model()
    
    if model_params:
        # Prediction Logic
        def predict_salary(profil, experience, niveau_etude, technologie, entreprise, model_params):
            theta = model_params['theta']
            mean = model_params['mean']
            std = model_params['std']
            feature_columns = model_params['feature_columns']
            
            input_data = pd.DataFrame({
                'Entreprise': [entreprise],
                'Profil': [profil],
                'Experience': [experience],
                'Niveau_Etude': [niveau_etude],
                'Technologie': [technologie],
                'Salaire': [0]
            })
            
            input_encoded = pd.get_dummies(input_data, columns=['Entreprise', 'Profil', 'Niveau_Etude', 'Technologie'], drop_first=False)
            input_encoded = input_encoded.drop('Salaire', axis=1)
            
            for col in feature_columns:
                if col not in input_encoded.columns:
                    input_encoded[col] = 0
            
            input_encoded = input_encoded[feature_columns]
            X_input = input_encoded.values
            X_input_norm = (X_input - mean) / std
            X_input_final = np.hstack((X_input_norm, np.ones((1, 1))))
            
            return X_input_final.dot(theta)[0][0]

        # UI
        col1, col2 = st.columns(2)
        with col1:
            profil = st.selectbox("Profil IT", sorted(model_params['profils_list']))
            experience = st.slider("Années d'expérience", 0, 15, 5)
            niveau_etude = st.selectbox("Niveau d'étude", sorted(model_params['niveaux_list']))
        
        with col2:
            technologie = st.selectbox("Technologie", sorted(model_params['technologies_list']))
            entreprise = st.selectbox("Entreprise", sorted(model_params['entreprises_list']))
            
        if st.button("Prédire le Salaire (Marouane)"):
            salaire = predict_salary(profil, experience, niveau_etude, technologie, entreprise, model_params)
            st.success(f"💰 Salaire Estimé: {salaire:,.2f} MAD")

# ==========================================
# 3. Salary Prediction (Asaad)
# ==========================================
def show_salary_prediction_asaad():
    st.title("Salary Predictor (Asaad)")
    
    model_path = '../other MLs/ML-Project-Asaad/AsaadModel/Asaad_Salaries_Models.pkl'
    
    @st.cache_resource
    def load_asaad_model():
        try:
            with open(model_path, "rb") as f:
                return pickle.load(f)
        except FileNotFoundError:
            st.error(f"Model file not found at {model_path}")
            return None

    data = load_asaad_model()
    
    if data:
        models = data["models"]
        month_map = data["month_map"]
        
        def predict(full_name, month, year):
            if full_name not in models:
                return None
            m = models[full_name]
            x = np.array([month_map[month], year])
            x = (x - m["X_mean"]) / m["X_std"]
            x = np.hstack([1, x]).reshape(1,-1)
            y_norm = x @ m["w"]
            y = float(y_norm * m["y_std"] + m["y_mean"])
            return y

        full_name = st.text_input("Enter Employee's full name :")
        month = st.text_input("Enter a valid month :")
        year = st.number_input("Enter a valid year :", min_value=2000, max_value=2100, value=2025)
        
        if st.button("Predict Salary (Asaad)"):
            if not full_name or not month:
                st.warning("Fill all form spots.")
            else:
                month = month.lower()
                if month not in month_map:
                    st.error("Invalid Month !")
                else:
                    salaire = predict(full_name, month, year)
                    if salaire is None:
                        st.error("Unknown Employee.")
                    else:
                        st.success(f"💵 Predicted Salary: {salaire:.2f} MAD")

# ==========================================
# Main Routing
# ==========================================
if app_mode == "Company Classification":
    show_company_classification()
elif app_mode == "Salary Prediction (Marouane)":
    show_salary_prediction_marouane()
elif app_mode == "Salary Prediction (Asaad)":
    show_salary_prediction_asaad()
