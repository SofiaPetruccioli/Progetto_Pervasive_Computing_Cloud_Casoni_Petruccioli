from google.cloud import firestore

# Inizializza Firestore dal file credenziali
db = firestore.Client.from_service_account_json('credentials.json')

# Definisce il documento di test
sensor_id = "TestRegion_TestCommodity"
new_entry = {
    'date': '2025-06-24',
    'commodity_name': 'TestCommodity',
    'state': 'TestRegion',
    'district': 'TestDistrict',
    'market': 'TestMarket',
    'min_price': 100,
    'max_price': 200,
    'modal_price': 150
}

# Inserisce o aggiorna il documento nella collection "commodities"
doc_ref = db.collection('commodities').document(sensor_id)
entity = doc_ref.get()

if entity.exists:
    d = entity.to_dict()
    d.setdefault('readings', []).append(new_entry)
    doc_ref.set(d)
    print(f"Aggiunto un nuovo reading a '{sensor_id}'")
else:
    doc_ref.set({
        'state': new_entry['state'],
        'readings': [new_entry]
    })
    print(f"Creato nuovo documento '{sensor_id}' con un reading")

print("✅ Scrittura completata. Ora puoi controllare Firestore.")
