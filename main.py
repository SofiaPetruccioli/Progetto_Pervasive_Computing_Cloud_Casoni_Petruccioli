from flask import Flask, render_template, request, redirect, jsonify
from flask_login import LoginManager, current_user, login_user, logout_user, login_required, UserMixin
from secret import secret_key
from google.cloud import firestore
from datetime import datetime
import json
from collections import defaultdict
from statistics import mean
import os
import joblib
import pandas as pd
from prophet import Prophet
from joblib import dump, load
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from google.cloud import storage
import tempfile
from google.oauth2 import service_account

# Percorso al file di credenziali
CREDENTIALS_PATH = "credentials.json"
credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)

app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key

login = LoginManager(app)
login.login_view = '/homepage'

db = firestore.Client.from_service_account_json('credentials.json')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username'].strip()
        password = request.form['password']
        role = request.form['role'].lower()

        user_ref = db.collection('users').document(username)
        if user_ref.get().exists:
            return render_template('register.html', message="❌ Username already exists.")

        user_ref.set({
            "username": username,
            "password": password,  # in chiaro come richiesto
            "role": role
        })

        return render_template('register.html', success=f"✅ User '{username}' registered successfully!")

    return render_template('register.html')


class User(UserMixin):
    def __init__(self, username, role='user'):
        self.id = username
        self.role = role



@login.user_loader
def load_user(username):
    doc = db.collection('users').document(username).get()
    if doc.exists:
        data = doc.to_dict()
        return User(username, role=data.get('role', 'user'))
    return None




@app.route('/')
@app.route('/homepage')
def home():
    return render_template('homepage.html')

@app.route('/login_admin', methods=['GET', 'POST'])
def login_admin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        doc = db.collection('users').document(username).get()
        if doc.exists:
            user = doc.to_dict()
            if user.get('password') == password and user.get('role') == 'admin':
                login_user(User(username, role='admin'))
                return redirect('/dashboard')
        
        return render_template('login_admin.html', error="Invalid credentials")
    
    return render_template('login_admin.html')


@app.route('/logout_admin')
@login_required
def logout_admin():
    logout_user()
    return render_template('homepage.html')

@app.route('/login_user', methods=['GET', 'POST'])
def user_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        doc = db.collection('users').document(username).get()
        if doc.exists:
            user = doc.to_dict()
            if user.get('password') == password and user.get('role') == 'user':
                login_user(User(username, role='user'))
                return redirect('/dashboard_user')

        return render_template('login_user.html', error="Invalid credentials")

    return render_template('login_user.html')


@app.route('/logout_user')
@login_required
def user_logout():
    logout_user()
    return render_template('homepage.html')

@app.route('/dashboard_user')
@login_required
def dashboard_user():
    return render_template('dashboard_user.html')

@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')

@app.route('/map')
@login_required
def map_page():
    return render_template('map.html')


@app.route('/graph3')
@login_required
def graph():
    docs = db.collection('commodities').stream()
    data_structure = defaultdict(lambda: {
        'states': defaultdict(lambda: defaultdict(list)),
        'districts': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        'markets': defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    })

    for doc in docs:
        content = doc.to_dict()
        for r in content.get('readings', []):
            try:
                date = datetime.strptime(r['date'], "%Y-%m-%d").date().isoformat()
                commodity = r['commodity_name']
                state = r['state']
                district = r['district']
                market = r['market']
                modal_price = float(r['modal_price'])

                data_structure[commodity]['states'][state][date].append(modal_price)
                data_structure[commodity]['districts'][state][district][date].append(modal_price)
                data_structure[commodity]['markets'][district][market][date].append(modal_price)

            except Exception as e:
                print("Error aggregation:", r.get('date'), "-", e)

    result = {}
    for commodity, levels in data_structure.items():
        result[commodity] = {
            'aggregated': [],
            'states': {},
            'districts': {},
            'markets': {}
        }

        aggregated_days = defaultdict(list)

        for state, dates in levels['states'].items():
            result[commodity]['states'][state] = [
                {'x': date, 'y': mean(prices)} for date, prices in sorted(dates.items())
            ]
            for date, prices in dates.items():
                aggregated_days[date].extend(prices)

        for state, districts in levels['districts'].items():
            result[commodity]['districts'][state] = {}
            for district, dates in districts.items():
                result[commodity]['districts'][state][district] = [
                    {'x': date, 'y': mean(prices)} for date, prices in sorted(dates.items())
                ]

        for district, markets in levels['markets'].items():
            result[commodity]['markets'][district] = {}
            for market, dates in markets.items():
                result[commodity]['markets'][district][market] = [
                    {'x': date, 'y': mean(prices)} for date, prices in sorted(dates.items())
                ]

        result[commodity]['aggregated'] = [
            {'x': date, 'y': mean(prices)} for date, prices in sorted(aggregated_days.items())
        ]

    return render_template('graph3.html', data=json.dumps(result))

@app.route('/database')
@login_required
def database():
    db = firestore.Client.from_service_account_json('credentials.json')
    docs = db.collection('commodities').stream()
    
    dati = []
    max_rows = 500

    for doc in docs:
        contenuto = doc.to_dict()
        readings = contenuto.get('readings', [])
        for r in readings:
            if len(dati) >= max_rows:
                break  # Ferma se abbiamo raggiunto 500 righe
            r['sensor'] = doc.id
            dati.append(r)
        if len(dati) >= max_rows:
            break  # Ferma anche l'iterazione sui documenti

    return render_template('database.html', dati=dati)

@app.route('/sensors/<sensor>', methods=['GET'])
@login_required
def read(sensor):
    entity = db.collection('commodities').document(sensor).get()
    if entity.exists:
        d = entity.to_dict()
        return json.dumps(d.get('readings', [])), 200
    else:
        return 'not found', 404

@app.route('/sensors/<sensor>', methods=['POST'])
def new_data(sensor):
    data = request.values
    new_entry = {
        'date': data['date'],
        'commodity_name': data['commodity_name'],
        'state': data['state'],
        'district': data['district'],
        'market': data['market'],
        'min_price': float(data['min_price']),
        'max_price': float(data['max_price']),
        'modal_price': float(data['modal_price']),
    }

    doc_ref = db.collection('commodities').document(sensor)
    entity = doc_ref.get()
    if entity.exists:
        d = entity.to_dict()
        d.setdefault('readings', []).append(new_entry)
        doc_ref.set(d)
    else:
        doc_ref.set({
            'state': new_entry['state'],
            'readings': [new_entry]
        })

    return 'ok', 200


@app.route('/getmapdata')
@login_required
def get_map_data():
    # Retrieve state coordinates from the state_coords collection
    state_coords_docs = db.collection('state_coords').stream()
    state_coords = {}
    for doc in state_coords_docs:
        data = doc.to_dict()
        state = data.get('state')
        lat = data.get('lat')
        lon = data.get('lon')
        if state and lat is not None and lon is not None:
            state_coords[state] = [lat, lon]

    docs = db.collection('commodities').stream()
    product_states = {}
    for doc in docs:
        content = doc.to_dict()
        for r in content.get('readings', []):
            prod = r['commodity_name']
            state = r['state']
            if prod not in product_states:
                product_states[prod] = set()
            product_states[prod].add(state)
    # Prepare output: { product: [ {state: name, coords: [lat,lon]}, ... ] }
    output = {}
    for prod, states in product_states.items():
        output[prod] = []
        for state in states:
            coords = state_coords.get(state)
            if coords:
                output[prod].append({'state': state, 'coords': coords})
    return app.response_class(
        response=json.dumps(output),
        status=200,
        mimetype='application/json'
    )

@app.route('/graph_map')
@login_required
def graph_map():
    product = request.args.get('product')
    state = request.args.get('state')

    # Struttura dati base per il risultato finale
    data_structure = defaultdict(lambda: {
        'states': defaultdict(lambda: defaultdict(list)),
        'districts': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        'markets': defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    })

    # Estrai e filtra i dati da Firestore
    docs = db.collection('commodities').stream()
    for doc in docs:
        content = doc.to_dict()
        for r in content.get('readings', []):
            try:
                # Filtro per prodotto e stato, se specificati
                if product and r.get('commodity_name') != product:
                    continue
                if state and r.get('state') != state:
                    continue

                date = pd.to_datetime(r['date']).strftime('%Y-%m-%d')
                commodity = r['commodity_name']
                state_ = r['state']
                district = r['district']
                market = r['market']
                modal_price = float(r['modal_price'])

                # Popola le strutture dati
                data_structure[commodity]['states'][state_][date].append(modal_price)
                data_structure[commodity]['districts'][state_][district][date].append(modal_price)
                data_structure[commodity]['markets'][district][market][date].append(modal_price)

            except Exception as e:
                print(f"Errore su record {r.get('date')}: {e}")

    # Costruisci il dizionario finale in formato richiesto da Chart.js
    result = {}
    for commodity, levels in data_structure.items():
        result[commodity] = {
            'aggregated': [],
            'states': {},
            'districts': {},
            'markets': {}
        }

        aggregated_days = defaultdict(list)

        # Livello stati
        for state_name, date_prices in levels['states'].items():
            result[commodity]['states'][state_name] = [
                {'x': date, 'y': mean(prices)} for date, prices in sorted(date_prices.items())
            ]
            for date, prices in date_prices.items():
                aggregated_days[date].extend(prices)

        # Livello distretti
        for state_name, district_map in levels['districts'].items():
            result[commodity]['districts'][state_name] = {}
            for district, date_prices in district_map.items():
                result[commodity]['districts'][state_name][district] = [
                    {'x': date, 'y': mean(prices)} for date, prices in sorted(date_prices.items())
                ]

        # Livello mercati
        for district_name, market_map in levels['markets'].items():
            result[commodity]['markets'][district_name] = {}
            for market, date_prices in market_map.items():
                result[commodity]['markets'][district_name][market] = [
                    {'x': date, 'y': mean(prices)} for date, prices in sorted(date_prices.items())
                ]

        # Livello aggregato
        result[commodity]['aggregated'] = [
            {'x': date, 'y': mean(prices)} for date, prices in sorted(aggregated_days.items())
        ]

    return render_template(
        'graph_map.html',
        data=json.dumps(result),
        selected_product=product,
        selected_state=state
    )




def upload_to_gcs(bucket_name, blob_name, local_path):
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(local_path)

def download_from_gcs(bucket_name, blob_name, local_path):
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.download_to_filename(local_path)

def model_exists_gcs(bucket_name, blob_name):
    client = storage.Client(credentials=credentials)
    bucket = client.bucket(bucket_name)
    return bucket.blob(blob_name).exists(client)



@app.route('/forecast', methods=['GET'])
@login_required
def forecast_page():
    product = request.args.get('product')
    all_products = set()
    records = []

    docs = db.collection('commodities').stream()
    for doc in docs:
        content = doc.to_dict()
        for r in content.get('readings', []):
            all_products.add(r['commodity_name'])
            if product and r['commodity_name'] == product:
                try:
                    r['date'] = pd.to_datetime(r['date'])
                    r['modal_price'] = float(r['modal_price'])
                    records.append(r)
                except:
                    continue

    if not product:
        return render_template('forecast.html', products=sorted(all_products), selected=None, data=None, message=None)

    df = pd.DataFrame(records)
    ts = df.groupby('date')['modal_price'].mean().reset_index()
    ts.rename(columns={'date': 'ds', 'modal_price': 'y'}, inplace=True)
    ts['floor'] = 0

    bucket_name = "my-forecast-models-bucket"
    blob_name = f"models/prophet_{product.replace(' ', '_')}.joblib"
    storage_client = storage.Client(credentials=credentials)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    if not blob.exists(storage_client):
        if len(ts) < 20:
            return render_template('forecast.html', products=sorted(all_products), selected=product,
                                   data=None, message=f"Insufficient number of samples for '{product}'")

        model = Prophet(
            growth='linear',
            seasonality_mode='additive',
            changepoint_prior_scale=0.03,
            seasonality_prior_scale=2
        )
        model.fit(ts)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmpfile:
            joblib.dump(model, tmpfile.name)
            blob.upload_from_filename(tmpfile.name)
    else:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmpfile:
            blob.download_to_filename(tmpfile.name)
            model = joblib.load(tmpfile.name)

    future = model.make_future_dataframe(periods=30)
    future['floor'] = 0
    forecast = model.predict(future)[['ds', 'yhat']].tail(30)
    forecast['tipo'] = 'previsione'
    forecast.rename(columns={'yhat': 'y'}, inplace=True)

    storico = ts[['ds', 'y']].copy()
    storico['tipo'] = 'storico'

    combined = pd.concat([storico, forecast])

    return render_template('forecast.html',
                           products=sorted(all_products),
                           selected=product,
                           data=combined.to_dict(orient='records'),
                           message=None)

from flask import Flask, jsonify




from flask import jsonify

@app.route('/aggiorna_modelli')
@login_required
def train_all_models(bucket_name="my-forecast-models-bucket", model_dir='models', min_points=35):
    os.makedirs(model_dir, exist_ok=True)

    # 1. Elimina tutti i file nella cartella models/
    for file in os.listdir(model_dir):
        file_path = os.path.join(model_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    try:
        # 2. Recupera i dati da Firestore
        docs = db.collection('commodities').stream()
        data = []
        for doc in docs:
            content = doc.to_dict()
            for r in content.get('readings', []):
                try:
                    r['commodity_name'] = r['commodity_name'].strip()
                    r['date'] = pd.to_datetime(r['date'])
                    r['modal_price'] = float(r['modal_price'])
                    data.append(r)
                except:
                    continue

        # 3. Costruisci modelli Prophet e salvali su GCS
        df = pd.DataFrame(data)
        for commodity in df['commodity_name'].unique():
            subset = df[df['commodity_name'] == commodity]
            ts = subset.groupby('date')['modal_price'].mean().reset_index()
            ts.rename(columns={'date': 'ds', 'modal_price': 'y'}, inplace=True)
            ts['floor'] = 0

            if len(ts) < min_points:
                continue

            model = Prophet(
                growth='linear',
                seasonality_mode='additive',
                changepoint_prior_scale=0.03,
                seasonality_prior_scale=2
            )
            model.fit(ts)

            # salva temporaneamente e carica su GCS
            with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
                dump(model, tmp.name)
                blob_name = f"models/prophet_{commodity.replace(' ', '_')}.joblib"
                upload_to_gcs(bucket_name, blob_name, tmp.name)

        return jsonify({"status": "success", "message": "Models updated successfully!"})

    except Exception as e:
        print("Error updating models:", str(e))
        return jsonify({"status": "error", "message": f"Error updating models: {str(e)}"}), 500
     


from flask import render_template, request
from flask_login import login_required
import pandas as pd

@app.route('/arbitrage_simple')
@login_required
def arbitrage_page():
    records = []

    # Recupera i dati dal database Firestore
    docs = db.collection('commodities').stream()
    for doc in docs:
        content = doc.to_dict()
        for r in content.get('readings', []):
            try:
                # Forza la data a stringa leggibile
                date_str = str(r['date']) if not isinstance(r['date'], str) else r['date']
                records.append({
                    'date': date_str,
                    'commodity_name': r['commodity_name'],
                    'market': r['market'],
                    'modal_price': float(r['modal_price'])
                })
            except Exception:
                continue  # ignora record malformati

    df = pd.DataFrame(records)

    if df.empty:
        return render_template(
            'arbitrage_simple.html',
            opportunities=[],
            products=[],
            selected_product=None
        )

    # Lista dei prodotti unici per il menu a tendina
    products = sorted(df['commodity_name'].dropna().unique().tolist())

    # Prodotto selezionato via GET, fallback sul primo disponibile
    selected_product = request.args.get('product')
    if selected_product not in products:
        selected_product = products[0] if products else None

    # Filtra i dati per il prodotto selezionato
    df = df[df['commodity_name'] == selected_product]

    # Calcola le migliori opportunità giornaliere
    opps = []
    for date, group in df.groupby('date'):
        if len(group) < 2:
            continue
        max_row = group.loc[group['modal_price'].idxmax()]
        min_row = group.loc[group['modal_price'].idxmin()]
        profit = max_row['modal_price'] - min_row['modal_price']
        profit_percent = (profit / min_row['modal_price']) * 100 if min_row['modal_price'] != 0 else 0

        opps.append({
            'date': date,
            'commodity_name': selected_product,
            'market_buy': min_row['market'],
            'price_buy': round(min_row['modal_price'], 2),
            'market_sell': max_row['market'],
            'price_sell': round(max_row['modal_price'], 2),
            'profit': round(profit, 2),
            'profit_percent': round(profit_percent, 2)
        })

    df_opps = pd.DataFrame(opps)

    # Ordina per data solo se ci sono dati
    if not df_opps.empty and 'date' in df_opps.columns:
        try:
            df_opps['date_obj'] = pd.to_datetime(df_opps['date'], errors='coerce', format='%Y-%m-%d')
            df_opps = df_opps.sort_values(by='date_obj').drop(columns=['date_obj'])
        except Exception:
            df_opps = df_opps.sort_values(by='date', errors='ignore')
    else:
        # assicura struttura per template
        df_opps = pd.DataFrame(columns=[
            'date', 'commodity_name', 'market_buy', 'price_buy',
            'market_sell', 'price_sell', 'profit', 'profit_percent'
        ])

    return render_template(
        'arbitrage_simple.html',
        opportunities=df_opps.to_dict(orient='records'),
        products=products,
        selected_product=selected_product
    )

@app.route('/arbitrage_chart')
@login_required
def arbitrage_chart():
    records = []

    # Estrai dati da Firestore
    docs = db.collection('commodities').stream()
    for doc in docs:
        content = doc.to_dict()
        for r in content.get('readings', []):
            try:
                records.append({
                    'date': str(r['date']),
                    'commodity_name': r['commodity_name'],
                    'market': r['market'],
                    'modal_price': float(r['modal_price'])
                })
            except:
                continue

    df = pd.DataFrame(records)
    if df.empty:
        return render_template('arbitrage_chart.html', products=[], selected_product=None, dates=[], profits=[])

    # Lista prodotti unici
    products = sorted(df['commodity_name'].dropna().unique().tolist())

    # Prodotto selezionato dalla query string
    selected_product = request.args.get('product') or products[0]

    if selected_product not in products:
        selected_product = products[0]

    # Filtro per prodotto
    df = df[df['commodity_name'] == selected_product]

    opps = []
    for date, group in df.groupby('date'):
        if len(group) < 2:
            continue
        try:
            max_price = group['modal_price'].max()
            min_price = group['modal_price'].min()
            profit = max_price - min_price
            opps.append({'date': date, 'profit': profit})
        except:
            continue

    df_opps = pd.DataFrame(opps)
    if df_opps.empty:
        return render_template('arbitrage_chart.html', products=products, selected_product=selected_product, dates=[], profits=[])

    df_opps['date'] = pd.to_datetime(df_opps['date'], errors='coerce')
    df_opps = df_opps.dropna().sort_values('date')
    df_opps['cum_profit'] = df_opps['profit'].cumsum()

    dates = df_opps['date'].dt.strftime('%Y-%m-%d').tolist()
    profits = df_opps['cum_profit'].round(2).tolist()

    return render_template(
        'arbitrage_chart.html',
        products=products,
        selected_product=selected_product,
        dates=dates,
        profits=profits
    )



if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)

