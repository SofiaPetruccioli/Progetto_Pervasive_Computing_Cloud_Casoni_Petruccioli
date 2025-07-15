import csv
from google.cloud import firestore

# Connessione a Firestore
db = firestore.Client.from_service_account_json('credentials.json')

CSV_PATH = 'data/agridata_clean.csv'
BULK_COLLECTION = 'bulk_commodities'  # Nuova collection per tenere separati i dati caricati manualmente

with open(CSV_PATH, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile)
    sensors_data = {}

    for row in reader:
        if any(value == '' or value is None for value in row.values()):
            continue

        sensor_id = f"{row['state'].replace(' ', '_')}_{row['commodity_name'].replace(' ', '_')}"
        entry = {
            'date': row['date'],
            'commodity_name': row['commodity_name'],
            'state': row['state'],
            'district': row['district'],
            'market': row['market'],
            'min_price': float(row['min_price']),
            'max_price': float(row['max_price']),
            'modal_price': float(row['modal_price']),
        }

        if sensor_id not in sensors_data:
            sensors_data[sensor_id] = {
                'state': row['state'],
                'readings': []
            }

        sensors_data[sensor_id]['readings'].append(entry)

# Scrive su Firestore nella collection separata
for sensor_id, doc in sensors_data.items():
    db.collection(BULK_COLLECTION).document(sensor_id).set(doc)

print(f"✅ Dati caricati su Firestore nella collection '{BULK_COLLECTION}'")
