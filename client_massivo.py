import csv
import requests
from concurrent.futures import ThreadPoolExecutor

# URL del server Flask per ricevere i dati
SERVER_URL = "http://localhost:8080/sensors"
# Percorso del file CSV originale con possibili caratteri \x00
CSV_PATH = "data/agridata_csv_202110311352.csv"
# Numero massimo di thread per inviare richieste in parallelo
MAX_WORKERS = 20

def clean_csv_content(file_path):
    """Legge e ripulisce il contenuto del CSV da caratteri nulli (\x00)."""
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        cleaned_lines = [line.replace('\x00', '') for line in f]
    return cleaned_lines

def send_row(row):
    """Invia una singola riga al server come dati sensore."""
    if any(v == '' for v in row.values()):
        return  # Salta le righe con valori vuoti
    sensor_id = f"{row['state'].replace(' ', '_')}_{row['commodity_name'].replace(' ', '_')}"
    data = {
        'date': row['date'],
        'commodity_name': row['commodity_name'],
        'state': row['state'],
        'district': row['district'],
        'market': row['market'],
        'min_price': row['min_price'],
        'max_price': row['max_price'],
        'modal_price': row['modal_price']
    }
    response = requests.post(f"{SERVER_URL}/{sensor_id}", data=data)
    print(f"Inviato: {row['commodity_name']} - Stato: {row['state']} - Risposta: {response.status_code}")

def main():
    # Pulizia del file CSV dai caratteri nulli
    cleaned_lines = clean_csv_content(CSV_PATH)
    reader = csv.DictReader(cleaned_lines)
    rows = list(reader)

    # Invia le righe al server in parallelo
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        executor.map(send_row, rows)

    print(">>> CLIENT MASSIVO COMPLETATO <<<")

if __name__ == "__main__":
    main()
