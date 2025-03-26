from django.shortcuts import render ,redirect
from django.http import HttpResponse
from .utils import get_plot
from .utils import X_Y
from .services import fetch_market_data
import random
import string

# Create your views here.
def connexion(request):
    return render(request,'connexion.html')

#def index(request):
#    return render(request,'index.html')

def index(request):
    data = fetch_market_data("AAPL")  # Récupère les données pour AAPL
    extracted_data = []
    if data and "data" in data:
        for item in data["data"]:
            produit_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
            extracted_data.append({
                "categorie": item.get("symbol"),
                "produit": produit_code,
                "open": item.get("open"),
                "close": item.get("close"),
                "volume": item.get("volume"),
            })
    return render(request, "index.html", {"market_data": extracted_data})

def contact(request):
    return render(request,'contact.html')

def service(request):
    return render(request,'service.html')

def graph(request):
    x,y =X_Y()
    chart = get_plot([48,49,50,51,52,53,54,55,56,57,58,59],x,y)
    rest=-y[59]+x[11]
    return render(request,'graph.html',{'chart': chart,'rest':rest})

def send(request):
    return render(request,'send.html')

def prod(request):
    return render(request,'prod.html')

def collecte(request):
    return render(request, 'collecte.html')

def map(request):
    return render(request, 'map.html')
def con(request):
    return render(request,'index.html')

def click(request):
    x,y =X_Y()
    chart = get_plot([48,49,50,51,52,53,54,55,56,57,58,59],x,y)
    rest=-y[59]+x[11]
    return render(request,'click.html',{'chart': chart,'rest':rest})

def market_data_view(request):
    data = fetch_market_data("AAPL")
    extracted_data = []
    if data and "data" in data:
        for item in data["data"]:
            extracted_data.append({
                "symbol": item.get("symbol"),
                "date": item.get("date"),
                "open": item.get("open"),
                "close": item.get("close"),
                "volume": item.get("volume"),
            })
    
    return render(request, "market_data.html", {"market_data": extracted_data})


def home_view(request):
    data = fetch_market_data("AAPL")  # On récupère les données pour AAPL
    extracted_data = []
    if data and "data" in data:
        for item in data["data"]:
            extracted_data.append({
                "symbol": item.get("symbol"),
                "date": item.get("date"),
                "open": item.get("open"),
                "close": item.get("close"),
                "volume": item.get("volume"),
            })

    return render(request, "index.html", {"market_data": extracted_data})