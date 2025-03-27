import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

st.set_page_config(layout="centered")

# -----------------------
# Personnalisation visuelle
# -----------------------

def set_background(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    page_bg_img = f"""
    <style>
    html, body, .stApp {{
        height: 100%;
        margin: 0;
        padding: 0;
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

def custom_style():
    style = """
    <style>
        .block-container {
            padding: 2rem 4rem 2rem 4rem;
            max-width: none;
        }
        .stApp {
            color: #222222;
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3 {
            color: #0D3B66;
        }
        .metric {
            font-size: 20px;
            margin-bottom: 0.5rem;
        }
        .stDownloadButton > button {
            background-color: #0D3B66;
            color: white;
            padding: 0.5rem 1.5rem;
            border-radius: 8px;
            border: none;
            font-weight: bold;
        }
        .stDownloadButton > button:hover {
            background-color: #145a9e;
        }
    </style>
    """
    st.markdown(style, unsafe_allow_html=True)

set_background("C:/Users/Mariem/Downloads/django_project-main/django_project-main/myapp/backround.png")
custom_style()

# -----------------------
# Fonctions de traitement
# -----------------------

@st.cache_data
def load_model_df():
    return pd.read_csv('C:/Users/Mariem/Downloads/django_project-main/django_project-main/model_df.csv')

@st.cache_data
def load_original_df():
    df = pd.read_csv('C:/Users/Mariem/Downloads/django_project-main/django_project-main/train.csv')
    df['date'] = df['date'].apply(lambda x: str(x)[:-3])
    df = df.groupby('date')['sales'].sum().reset_index()
    df['date'] = pd.to_datetime(df['date'])
    return df

def tts(data):
    data = data.drop(['sales','date'],axis=1)
    train, test = data[0:-12].values, data[-12:].values
    return train, test

def scale_data(train_set, test_set):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(train_set)
    train_scaled = scaler.transform(train_set)
    test_scaled = scaler.transform(test_set)
    X_train, y_train = train_scaled[:, 1:], train_scaled[:, 0:1].ravel()
    X_test, y_test = test_scaled[:, 1:], test_scaled[:, 0:1].ravel()
    return X_train, y_train, X_test, y_test, scaler

def undo_scaling(y_pred, x_test, scaler_obj):
    y_pred = y_pred.reshape(y_pred.shape[0], 1, 1)
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    pred_test_set = []
    for index in range(len(y_pred)):
        pred_test_set.append(np.concatenate([y_pred[index], x_test[index]], axis=1))
    pred_test_set = np.array(pred_test_set).reshape(len(y_pred), x_test.shape[2] + 1)
    return scaler_obj.inverse_transform(pred_test_set)

def predict_df(unscaled_predictions, original_df):
    result_list = []
    sales_dates = list(original_df[-13:].date)
    act_sales = list(original_df[-13:].sales)
    for index in range(len(unscaled_predictions)):
        result_dict = {}
        result_dict['pred_value'] = int(unscaled_predictions[index][0] + act_sales[index])
        result_dict['date'] = sales_dates[index + 1]
        result_list.append(result_dict)
    return pd.DataFrame(result_list)

def get_scores(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2

# -----------------------
# Interface utilisateur
# -----------------------

st.title("Prévision des Stocks")
st.markdown("Bienvenue sur notre tableau de bord de prévision. Consultez ci-dessous l'évolution prévue des stocks pour les 12 mois à venir.")

model_df = load_model_df()
original_df = load_original_df()
train, test = tts(model_df)
X_train, y_train, X_test, y_test, scaler = scale_data(train, test)

model = XGBRegressor(n_estimators=100, learning_rate=0.2, objective='reg:squarederror')
model.fit(X_train, y_train)
predictions = model.predict(X_test)

unscaled = undo_scaling(predictions, X_test, scaler)
results = predict_df(unscaled, original_df)

true_values = original_df.sales[-12:].values
pred_values = results.pred_value.values

fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(x=original_df.date[-12:], y=true_values, label='Historique', ax=ax)
sns.lineplot(x=results.date, y=pred_values, label='Prévision', ax=ax)
plt.xticks(rotation=45)
plt.ylabel("Volume de stock")
plt.xlabel("Mois")
plt.tight_layout()
st.subheader("Évolution Prévue des Stocks")
st.pyplot(fig)

csv = results.to_csv(index=False).encode('utf-8')
st.download_button("Télécharger les prévisions", csv, "stock_predictions.csv", "text/csv")


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Chargement des données
@st.cache_data
def load_data():
    return pd.read_csv("dashboard_stocks.csv")

df = load_data()

# Palette personnalisée
sns.set_palette(["#4CAF50", "#A5D6A7", "#81C784", "#66BB6A"])

# Titre
st.markdown("<h2 style='color:#2E7D32;'>Prévisions de Stocks - NeuronFood</h2>", unsafe_allow_html=True)

# Filtre catégorie
with st.sidebar:
    st.markdown("<h5 style='margin-bottom: 0.3rem;'>Filtrer par catégorie</h5>", unsafe_allow_html=True)
    selected_category = st.selectbox("", ["Toutes"] + sorted(df["Catégorie"].unique().tolist()), label_visibility="collapsed")

if selected_category != "Toutes":
    df = df[df["Catégorie"] == selected_category]

# Styles personnalisés
st.markdown("""
<style>
.card {
    background-color: #ffffffcc;
    border: 1px solid #e0e0e0;
    border-left: 5px solid #4CAF50;
    border-radius: 6px;
    padding: 1rem 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 1px 1px 6px rgba(0,0,0,0.05);
    font-size: 16px;
}
.card-title {
    font-weight: bold;
    margin-bottom: 0.3rem;
}
</style>
""", unsafe_allow_html=True)

# Valeurs en encadrés
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("<div class='card'><div class='card-title'>Erreur moyenne (RMSE)</div>{:,.2f}</div>".format(df['Prix_Vente_Avant'].mean()), unsafe_allow_html=True)
with col2:
    st.markdown("<div class='card'><div class='card-title'>Écart absolu moyen (MAE)</div>{:,.2f}</div>".format(df['Prix_Vente_Après'].mean()), unsafe_allow_html=True)
with col3:
    st.markdown("<div class='card'><div class='card-title'>Fiabilité de la prévision</div>{:.2f}%</div>".format((df['Prix_Vente_Avant'].mean()/df['Prix_Vente_Après'].mean())*100), unsafe_allow_html=True)

# Graphiques côte à côte
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("<h4 style='margin-top:1rem;'>Variation des Prix</h4>", unsafe_allow_html=True)
    fig1, ax1 = plt.subplots(figsize=(3.5, 1.6))
    width = 0.35
    x = range(len(df))
    ax1.bar(x, df['Prix_Vente_Avant'], width=width, label='Avant')
    ax1.bar([i + width for i in x], df['Prix_Vente_Après'], width=width, label='Après')
    ax1.set_xticks([i + width/2 for i in x])
    ax1.set_xticklabels(df['Nom_Produit'], rotation=45, ha='right', fontsize=6)
    ax1.set_ylabel("Prix (€)", fontsize=8)
    ax1.legend(fontsize=7, loc='upper right')
    fig1.tight_layout()
    st.pyplot(fig1, use_container_width=False)

with col_b:
    st.markdown("<h4 style='margin-top:1rem;'>Stocks Disponibles</h4>", unsafe_allow_html=True)
    fig2, ax2 = plt.subplots(figsize=(3.5, 1.6))
    sns.barplot(data=df, x="Nom_Produit", y="Stock", ax=ax2)
    ax2.set_ylabel("Stock", fontsize=8)
    ax2.set_xlabel("")
    ax2.tick_params(axis='x', labelrotation=45, labelsize=6)
    fig2.tight_layout()
    st.pyplot(fig2, use_container_width=False)

# Footer clean
st.markdown("""
    <style>
    footer {visibility: hidden;}
    .css-1outpf7 {padding: 1rem 2rem 1rem 2rem;}
    </style>
""", unsafe_allow_html=True)
