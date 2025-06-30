import csv
import requests
import io

SERVER_URL = "http://localhost:8080/sensors"
CSV_PATH = "data/agridata_clean.csv"

def main():
    with open(CSV_PATH, 'rb') as f:
        content = f.read().replace(b'\x00', b'')  # Rimuove i null byte
    text = content.decode('utf-8', errors='ignore')  # Decodifica ignorando eventuali errori
    reader = csv.DictReader(io.StringIO(text))

    for row in reader:
        if any(value == '' or value is None for value in row.values()):
            continue  # Salta righe con campi vuoti

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

if __name__ == "__main__":
    main()
    print(">>> CLIENT COMPLETATO <<<")
