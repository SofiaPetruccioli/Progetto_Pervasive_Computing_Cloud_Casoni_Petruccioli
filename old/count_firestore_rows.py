
from google.cloud import firestore

# Inizializza il client Firestore usando il file di credenziali
db = firestore.Client.from_service_account_json('credentials.json')

# Funzione per contare tutte le letture nel database
def count_firestore_readings():
    total_readings = 0
    docs = db.collection('commodities').stream()
    for doc in docs:
        data = doc.to_dict()
        readings = data.get('readings', [])
        total_readings += len(readings)
    return total_readings

# Esegui la funzione e stampa il risultato
total = count_firestore_readings()
print(f"Totale righe nel database Firestore: {total}")
