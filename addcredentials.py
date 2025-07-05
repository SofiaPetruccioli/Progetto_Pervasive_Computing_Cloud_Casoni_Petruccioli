from google.cloud import firestore

# Inizializza il client Firestore con le credenziali
db = firestore.Client.from_service_account_json('credentials.json')

# Dati dell'utente admin da inserire
username = "admin"
password = "admin123"
role = "admin"

# Documento Firestore (con ID = username)
user_ref = db.collection('users').document(username)

# Verifica se esiste già
if user_ref.get().exists:
    print(f"⚠️ L'utente '{username}' esiste già nel database.")
else:
    user_ref.set({
        "username": username,
        "password": password,
        "role": role
    })
    print(f"✅ Utente admin '{username}' inserito correttamente.")
