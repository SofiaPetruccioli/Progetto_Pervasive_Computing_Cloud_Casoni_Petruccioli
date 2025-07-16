#librerie
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

# Path to the credentials file
CREDENTIALS_PATH = "credentials.json"
credentials = service_account.Credentials.from_service_account_file(CREDENTIALS_PATH)

app = Flask(__name__)
app.config['SECRET_KEY'] = secret_key

#login manager
login = LoginManager(app)
login.login_view = '/homepage'

db = firestore.Client.from_service_account_json('credentials.json')




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

    doc_ref = db.collection('comm').document(sensor)
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
            "password": password,  
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

#login and logout
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

#dashboard user
@app.route('/dashboard_user')
@login_required
def dashboard_user():
    return render_template('dashboard_user.html')
#dashboard admin
@app.route('/dashboard')
@login_required
def dashboard():
    return render_template('dashboard.html')
#map
@app.route('/map')
@login_required
def map_page():
    return render_template('map.html')

#graph
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

#client side
@app.route('/sensors/<sensor>', methods=['GET'])
@login_required
def read(sensor):
    entity = db.collection('commodities').document(sensor).get()
    if entity.exists:
        d = entity.to_dict()
        return json.dumps(d.get('readings', [])), 200
    else:
        return 'not found', 404


#get map data
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

#graph of the map
@app.route('/graph_map')
@login_required
def graph_map():
    product = request.args.get('product')
    state = request.args.get('state')

    # base data structure for the final result
    data_structure = defaultdict(lambda: {
        'states': defaultdict(lambda: defaultdict(list)),
        'districts': defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        'markets': defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    })

    # Extract and filter data from Firestore
    docs = db.collection('commodities').stream()
    for doc in docs:
        content = doc.to_dict()
        for r in content.get('readings', []):
            try:
                # Filter for product and state, if specified
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

                # Populate the data structures
                data_structure[commodity]['states'][state_][date].append(modal_price)
                data_structure[commodity]['districts'][state_][district][date].append(modal_price)
                data_structure[commodity]['markets'][district][market][date].append(modal_price)

            except Exception as e:
                print(f"Errore su record {r.get('date')}: {e}")

    # Build the final dictionary in the format required by Chart.js
    result = {}
    for commodity, levels in data_structure.items():
        result[commodity] = {
            'aggregated': [],
            'states': {},
            'districts': {},
            'markets': {}
        }

        aggregated_days = defaultdict(list)

        # level states
        for state_name, date_prices in levels['states'].items():
            result[commodity]['states'][state_name] = [
                {'x': date, 'y': mean(prices)} for date, prices in sorted(date_prices.items())
            ]
            for date, prices in date_prices.items():
                aggregated_days[date].extend(prices)

        # level districts
        for state_name, district_map in levels['districts'].items():
            result[commodity]['districts'][state_name] = {}
            for district, date_prices in district_map.items():
                result[commodity]['districts'][state_name][district] = [
                    {'x': date, 'y': mean(prices)} for date, prices in sorted(date_prices.items())
                ]

        # level markets
        for district_name, market_map in levels['markets'].items():
            result[commodity]['markets'][district_name] = {}
            for market, date_prices in market_map.items():
                result[commodity]['markets'][district_name][market] = [
                    {'x': date, 'y': mean(prices)} for date, prices in sorted(date_prices.items())
                ]

        # aggregated level
        result[commodity]['aggregated'] = [
            {'x': date, 'y': mean(prices)} for date, prices in sorted(aggregated_days.items())
        ]

    return render_template(
        'graph_map.html',
        data=json.dumps(result),
        selected_product=product,
        selected_state=state
    )


# upload and download from gcs

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


#get data for the forecast
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
        if len(ts) < 35:
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


#get data for the forecast
@app.route('/forecast_user', methods=['GET'])
@login_required
def forecast_page_user():
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
        return render_template('forecast_user.html', products=sorted(all_products), selected=None, data=None, message=None)

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
        if len(ts) < 35:
            return render_template('forecast_user.html', products=sorted(all_products), selected=product,
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

    return render_template('forecast_user.html',
                           products=sorted(all_products),
                           selected=product,
                           data=combined.to_dict(orient='records'),
                           message=None)

#update forecast models
@app.route('/aggiorna_modelli')
@login_required
def train_all_models(bucket_name="my-forecast-models-bucket", model_dir='models', min_points=35):
    os.makedirs(model_dir, exist_ok=True)

    # 1. Delete all files in the models/ directory
    for file in os.listdir(model_dir):
        file_path = os.path.join(model_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    try:
        # 2. Get data from Firestore
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

        # 3. Build Prophet models and save them to GCS
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

            # save temporarily and upload to GCS
            with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp:
                dump(model, tmp.name)
                blob_name = f"models/prophet_{commodity.replace(' ', '_')}.joblib"
                upload_to_gcs(bucket_name, blob_name, tmp.name)

        return jsonify({"status": "success", "message": "Models updated successfully!"})

    except Exception as e:
        print("Error updating models:", str(e))
        return jsonify({"status": "error", "message": f"Error updating models: {str(e)}"}), 500
     

#Daily arbitrage opportunities
@app.route('/arbitrage_simple')
@login_required
def arbitrage_page():
    records = []

    # Get data from Firestore
    docs = db.collection('commodities').stream()
    for doc in docs:
        content = doc.to_dict()
        for r in content.get('readings', []):
            try:
                # Force the date to a readable string
                date_str = str(r['date']) if not isinstance(r['date'], str) else r['date']
                records.append({
                    'date': date_str,
                    'commodity_name': r['commodity_name'],
                    'market': r['market'],
                    'modal_price': float(r['modal_price'])
                })
            except Exception:
                continue  # ignore malformed records

    df = pd.DataFrame(records)

    if df.empty:
        return render_template(
            'arbitrage_simple.html',
            opportunities=[],
            products=[],
            selected_product=None
        )

    # unique products for the dropdown menu
    products = sorted(df['commodity_name'].dropna().unique().tolist())

    # selected product via GET, fallback on the first available
    selected_product = request.args.get('product')
    if selected_product not in products:
        selected_product = products[0] if products else None

    # Filter data for the selected product
    df = df[df['commodity_name'] == selected_product]

    # Calculate the best daily opportunities
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

    # sort by date only if there are data
    if not df_opps.empty and 'date' in df_opps.columns:
        try:
            df_opps['date_obj'] = pd.to_datetime(df_opps['date'], errors='coerce', format='%Y-%m-%d')
            df_opps = df_opps.sort_values(by='date_obj').drop(columns=['date_obj'])
        except Exception:
            df_opps = df_opps.sort_values(by='date', errors='ignore')
    else:
        # ensure structure for template
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


#Daily arbitrage opportunities
@app.route('/arbitrage_simple_user')
@login_required
def arbitrage_page_user():
    records = []

    # Get data from Firestore
    docs = db.collection('commodities').stream()
    for doc in docs:
        content = doc.to_dict()
        for r in content.get('readings', []):
            try:
                # Force the date to a readable string
                date_str = str(r['date']) if not isinstance(r['date'], str) else r['date']
                records.append({
                    'date': date_str,
                    'commodity_name': r['commodity_name'],
                    'market': r['market'],
                    'modal_price': float(r['modal_price'])
                })
            except Exception:
                continue  # ignore malformed records

    df = pd.DataFrame(records)

    if df.empty:
        return render_template(
            'arbitrage_simple_user.html',
            opportunities=[],
            products=[],
            selected_product=None
        )

    # unique products for the dropdown menu
    products = sorted(df['commodity_name'].dropna().unique().tolist())

    # selected product via GET, fallback on the first available
    selected_product = request.args.get('product')
    if selected_product not in products:
        selected_product = products[0] if products else None

    # Filter data for the selected product
    df = df[df['commodity_name'] == selected_product]

    # Calculate the best daily opportunities
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

    # sort by date only if there are data
    if not df_opps.empty and 'date' in df_opps.columns:
        try:
            df_opps['date_obj'] = pd.to_datetime(df_opps['date'], errors='coerce', format='%Y-%m-%d')
            df_opps = df_opps.sort_values(by='date_obj').drop(columns=['date_obj'])
        except Exception:
            df_opps = df_opps.sort_values(by='date', errors='ignore')
    else:
        # ensure structure for template
        df_opps = pd.DataFrame(columns=[
            'date', 'commodity_name', 'market_buy', 'price_buy',
            'market_sell', 'price_sell', 'profit', 'profit_percent'
        ])

    return render_template(
        'arbitrage_simple_user.html',
        opportunities=df_opps.to_dict(orient='records'),
        products=products,
        selected_product=selected_product
    )






# Arbitrage profit chart 
@app.route('/arbitrage_chart')
@login_required
def arbitrage_chart():
    records = []

    # Get data from Firestore
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

    # unique products for the dropdown menu
    products = sorted(df['commodity_name'].dropna().unique().tolist())

    # selected product via GET, fallback on the first available
    selected_product = request.args.get('product') or products[0]

    if selected_product not in products:
        selected_product = products[0]

    # Filter for product
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




#Arbitrage opportunities ML model: Random Forest
def train_arbitrage_model(model_dir='models'):
    # Extract all products
    docs = db.collection('commodities').stream()
    all_products = set()
    records_by_product = {}
    for doc in docs:
        content = doc.to_dict()
        for r in content.get('readings', []):
            prod = r['commodity_name']
            try:
                rec = {
                    'date': r['date'],
                    'commodity_name': prod,
                    'state': r['state'],
                    'district': r['district'],
                    'market': r['market'],
                    'min_price': float(r['min_price']),
                    'max_price': float(r['max_price']),
                    'modal_price': float(r['modal_price'])
                }
                records_by_product.setdefault(prod, []).append(rec)
            except Exception:
                continue
    os.makedirs(model_dir, exist_ok=True)
    pattern_stats_dict = {}
    for product, records in records_by_product.items():
        print(f"Product: {product}, total records: {len(records)}")
        df = pd.DataFrame(records)
        pairs = []
        for (product, date), group in df.groupby(['commodity_name', 'date']):
            markets = group['market'].tolist()
            prices = group['modal_price'].tolist()
            min_prices = group['min_price'].tolist()
            max_prices = group['max_price'].tolist()
            for i, j in combinations(range(len(markets)), 2):
                price1, price2 = prices[i], prices[j]
                min1, max1 = min_prices[i], max_prices[i]
                min2, max2 = min_prices[j], max_prices[j]
                diff = price2 - price1
                ratio = price2 / price1 if price1 != 0 else 0
                arbitrage = 1 if abs(diff) / min(price1, price2) > 0.1 else 0
                profit = abs(diff)
                profit_percent = profit / min(price1, price2) * 100 if min(price1, price2) != 0 else 0
                pairs.append({
                    'commodity_name': product,
                    'date': date,
                    'market1': markets[i],
                    'market2': markets[j],
                    'modal_price1': price1,
                    'modal_price2': price2,
                    'min_price1': min1,
                    'max_price1': max1,
                    'min_price2': min2,
                    'max_price2': max2,
                    'diff': diff,
                    'ratio': ratio,
                    'arbitrage': arbitrage,
                    'profit': profit,
                    'profit_percent': profit_percent
                })
        print(f"  Market pairs generated: {len(pairs)}")
        df_pairs = pd.DataFrame(pairs)
        if df_pairs.empty:
            print(f"  No valid market pairs for {product}, model NOT created.")
            continue
        if len(df_pairs) < 2:
            print(f"  Not enough data for {product} (only {len(df_pairs)} pairs), model NOT created.")
            continue
        feature_cols = ['diff', 'ratio', 'modal_price1', 'modal_price2', 'min_price1', 'max_price1', 'min_price2', 'max_price2']
        X = df_pairs[feature_cols]
        y = df_pairs['arbitrage']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        df_pairs['arbitrage_pred'] = clf.predict(X)
        pattern_stats = (
            df_pairs
            .groupby(['market1', 'market2'])
            .agg(
                arbitrage_count=('arbitrage_pred', 'sum'),
                avg_profit=('profit', 'mean'),
                avg_profit_percent=('profit_percent', 'mean'),
                total_obs=('arbitrage_pred', 'count'),
                avg_price1=('modal_price1', 'mean'),
                avg_price2=('modal_price2', 'mean')
            )
            .reset_index()
        )
        pattern_stats['arbitrage_freq'] = pattern_stats['arbitrage_count'] / pattern_stats['total_obs']
        # Save the model and stats to GCS
        import tempfile
        from joblib import dump
        bucket_name = "my-forecast-models-bucket"
        # Save model to temp file and upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_model:
            dump(clf, tmp_model.name)
            blob_name = f"arbitrage/arbitrage_rf_{product.replace(' ', '_')}.joblib"
            upload_to_gcs(bucket_name, blob_name, tmp_model.name)
        # Save stats to temp file and upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_stats:
            pattern_stats.to_csv(tmp_stats.name, index=False)
            blob_name = f"arbitrage/pattern_stats_{product.replace(' ', '_')}.csv"
            upload_to_gcs(bucket_name, blob_name, tmp_stats.name)
        pattern_stats_dict[product] = pattern_stats
        print(f"  ML model and stats saved to GCS for {product}")
    return pattern_stats_dict

#view pattern stats with ML model
@app.route('/pattern_stats')
@login_required
def pattern_stats_page():
    product = request.args.get('product')
    # Get all products like in arbitrage_simple
    records = []
    docs = db.collection('commodities').stream()
    for doc in docs:
        content = doc.to_dict()
        for r in content.get('readings', []):
            try:
                records.append({
                    'commodity_name': r['commodity_name']
                })
            except Exception:
                continue
    df = pd.DataFrame(records)
    products = sorted(df['commodity_name'].dropna().unique().tolist())
    if not product or product not in products:
        product = products[0] if products else None
    # Download the model and pattern_stats for the selected product from GCS
    import tempfile
    bucket_name = "my-forecast-models-bucket"
    model_blob = f"arbitrage/arbitrage_rf_{product.replace(' ', '_')}.joblib"
    stats_blob = f"arbitrage/pattern_stats_{product.replace(' ', '_')}.csv"
    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_model:
        try:
            download_from_gcs(bucket_name, model_blob, tmp_model.name)
        except Exception:
            return f"No arbitrage model found for product {product}", 404
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp_stats:
        try:
            download_from_gcs(bucket_name, stats_blob, tmp_stats.name)
        except Exception:
            return f"No pattern stats found for product {product}. Please update the models.", 404
        df_stats = pd.read_csv(tmp_stats.name)
    # Filter: only patterns with frequency >= 0.8 and show only the first 5
    df_stats = df_stats[df_stats['arbitrage_freq'] >= 0.8]
    pattern_stats_table = df_stats.sort_values('total_obs', ascending=False).to_dict(orient='records')
    pattern_stats_table = pattern_stats_table[:5] 
    return render_template('pattern_stats.html', pattern_stats=pattern_stats_table, selected_product=product, products=products)

#retrain arbitrage models: call the function train_arbitrage_model()
@app.route('/retrain_arbitrage_models', methods=['POST'])
@login_required
def retrain_arbitrage_models():
    try:
        train_arbitrage_model()
        return jsonify({'status': 'success', 'message': 'Arbitrage models retrained and saved'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Error during retraining: {str(e)}'}), 500

if __name__ == '__main__':
  port=int(os.environ.get('PORT', 8080))
  app.run(host="0.0.0.0", port=port, debug=True)

