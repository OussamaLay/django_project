# stocks/services.py

import requests

# Définissez votre clé d'accès à l'API
API_KEY = "a5324289a5b3d5b8cac489922e91e461"  # Remplacez par votre clé réelle

# URL de base de l'endpoint EOD
API_URL = "http://api.marketstack.com/v1/eod"

def fetch_market_data(symbols="AAPL"):
    """
    Cette fonction effectue une requête à l'API Marketstack pour récupérer
    les données de fin de journée (EOD) pour le ou les ticker(s) passé(s) en paramètre.
    """
    # Paramètres de la requête
    params = {
        "access_key": API_KEY,
        "symbols": symbols,
        "limit": 10
    }
    
    try:
        # Effectuer la requête GET
        response = requests.get(API_URL, params=params)
        response.raise_for_status()  # Vérifie les erreurs HTTP
        
        # Parser la réponse JSON
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        # Gérer les exceptions en les loggant ou en les renvoyant
        print("Erreur lors de la requête API:", e)
        return None
