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
from joblib import dump
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


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
        # Approximate centroids of Indian states (name: [lat, lon])
    state_coords = {
        'Andhra Pradesh': [15.9129, 79.74],
        'Arunachal Pradesh': [28.218, 94.7278],
        'Assam': [26.2006, 92.9376],
        'Bihar': [25.0961, 85.3131],
        'Chhattisgarh': [21.2787, 81.8661],
        'Goa': [15.2993, 74.124],
        'Gujarat': [22.2587, 71.1924],
        'Haryana': [29.0588, 76.0856],
        'Himachal Pradesh': [31.1048, 77.1734],
        'Jharkhand': [23.6102, 85.2799],
        'Karnataka': [15.3173, 75.7139],
        'Kerala': [10.8505, 76.2711],
        'Madhya Pradesh': [22.9734, 78.6569],
        'Maharashtra': [19.7515, 75.7139],
        'Manipur': [24.6637, 93.9063],
        'Meghalaya': [25.467, 91.3662],
        'Mizoram': [23.1645, 92.9376],
        'Nagaland': [26.1584, 94.5624],
        'Odisha': [20.9517, 85.0985],
        'Punjab': [31.1471, 75.3412],
        'Rajasthan': [27.0238, 74.2179],
        'Sikkim': [27.533, 88.5122],
        'Tamil Nadu': [11.1271, 78.6569],
        'Telangana': [18.1124, 79.0193],
        'Tripura': [23.9408, 91.9882],
        'Uttar Pradesh': [26.8467, 80.9462],
        'Uttarakhand': [30.0668, 79.0193],
        'West Bengal': [22.9868, 87.855],
        'Delhi': [28.7041, 77.1025],
        'Jammu and Kashmir': [33.7782, 76.5762],
        'Ladakh': [34.1526, 77.5771],
    }
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
                if r['commodity_name'] == product and r['state'] == state:
                    date = datetime.strptime(r['date'], "%Y-%m-%d").date().isoformat()
                    commodity = r['commodity_name']
                    state_ = r['state']
                    district = r['district']
                    market = r['market']
                    modal_price = float(r['modal_price'])
                    data_structure[commodity]['states'][state_][date].append(modal_price)
                    data_structure[commodity]['districts'][state_][district][date].append(modal_price)
                    data_structure[commodity]['markets'][district][market][date].append(modal_price)
            except Exception as e:
                print("Errore aggregazione:", r.get('date'), "-", e)
    result = {}
    for commodity, levels in data_structure.items():
        result[commodity] = {
            'states': {},
            'districts': {},
            'markets': {}
        }
        for state, dates in levels['states'].items():
            result[commodity]['states'][state] = [
                {'x': date, 'y': mean(prices)} for date, prices in sorted(dates.items())
            ]
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
    return render_template('graph_map.html', data=json.dumps(result))




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
    ts['floor'] = 0  # imposta limite inferiore per il training

    model_path = f"models/prophet_{product.replace(' ', '_')}.joblib"
    if not os.path.exists(model_path):
        if len(ts) < 20:
            return render_template('forecast.html', products=sorted(all_products), selected=product,
                                   data=None, message=f"Insufficient number of sample for '{product}'")
        model = Prophet(
            growth='linear',
            seasonality_mode='additive',
            changepoint_prior_scale=0.03,
            seasonality_prior_scale=3
        )
        model.fit(ts)
        joblib.dump(model, model_path)
    else:
        model = joblib.load(model_path)

    future = model.make_future_dataframe(periods=30)
    future['floor'] = 0  # imposta limite inferiore anche per le previsionivi
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




@app.route('/aggiorna_modelli')
def aggiorna_modelli():
    train_all_models()  # Chiama la funzione
    return jsonify({'message': 'Modelli aggiornati con successo'})  # Risposta JSON al client

def train_all_models(model_dir='models', min_points=20):
    os.makedirs(model_dir, exist_ok=True)
    docs = db.collection('commodities').stream()
    data = []

    for doc in docs:
        content = doc.to_dict()
        for r in content.get('readings', []):
            try:
                r['commodity_name'] = r['commodity_name'].strip()
                r['date'] = pd.to_datetime(r['date'])
                r['modal_price'] = float(r['modal_price'])
                r['floor'] = 0
                data.append(r)
            except:
                continue

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
            seasonality_prior_scale=3
        )
        model.fit(ts)
        path = os.path.join(model_dir, f"prophet_{commodity.replace(' ', '_')}.joblib")
        dump(model, path)




def train_arbitrage_model():
    selected_products = ["Wheat", "Rice", "Carrots"]
    records = []
    docs = db.collection('commodities').stream()
    for doc in docs:
        content = doc.to_dict()
        for r in content.get('readings', []):
            try:
                if r['commodity_name'] not in selected_products:
                    continue
                records.append({
                    'date': r['date'],
                    'commodity_name': r['commodity_name'],
                    'state': r['state'],
                    'district': r['district'],
                    'market': r['market'],
                    'min_price': float(r['min_price']),
                    'max_price': float(r['max_price']),
                    'modal_price': float(r['modal_price'])
                })
            except Exception:
                continue
    df = pd.DataFrame(records)
    print(f"Total records: {len(df)}")
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
    df_pairs = pd.DataFrame(pairs)
    print(f"Total pairs: {len(df_pairs)}")
    #debug
    print("Sample of arbitrage dataset (first 20 rows):")
    print(df_pairs[['commodity_name','date','market1','modal_price1','market2','modal_price2','diff','arbitrage','profit','profit_percent']].head(20))
    #end debug
    feature_cols = ['diff', 'ratio', 'modal_price1', 'modal_price2', 'min_price1', 'max_price1', 'min_price2', 'max_price2']
    X = df_pairs[feature_cols]
    y = df_pairs['arbitrage']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    print("Accuracy su test set:", clf.score(X_test, y_test))
    dump(clf, 'models/arbitrage_rf.joblib')
    opps = df_pairs[(df_pairs['arbitrage'] == 1) & (df_pairs['profit'] > 0)]
    print(opps[['commodity_name', 'date', 'market1', 'modal_price1', 'market2', 'modal_price2', 'profit', 'profit_percent']].sort_values('profit_percent', ascending=False).head(10))
    return clf, df_pairs

@app.route('/arbitrage')
@login_required
def arbitrage_page():
    _, df_pairs = train_arbitrage_model()
    opps = df_pairs[(df_pairs['arbitrage'] == 1) & (df_pairs['profit'] > 0)]
    top_opps = opps.sort_values('profit_percent', ascending=False).head(50)
    opportunities = top_opps.to_dict(orient='records')
    return render_template('arbitrage.html', opportunities=opportunities)

if __name__ == '__main__':
    

    app.run(host="0.0.0.0", port=8080, debug=True)

