
from google.cloud import firestore

db = firestore.Client.from_service_account_json('credentials.json')
collection_ref = db.collection('commodities')

def delete_all_documents():
    docs = collection_ref.stream()
    for doc in docs:
        print(f"Eliminazione documento: {doc.id}")
        doc.reference.delete()

if __name__ == "__main__":
    delete_all_documents()
    print(">>> TUTTI I DOCUMENTI SONO STATI ELIMINATI <<<")
