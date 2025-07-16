import csv
import requests
import io
import time

SERVER_URL = "https://pervasive-api-1085959941354.europe-west1.run.app/sensors"
CSV_PATH = "data/agridata_clean.csv"
DELAY_SECONDS = 2  # interval between requests

def main():
    with open(CSV_PATH, 'rb') as f:
        content = f.read().replace(b'\x00', b'')
    text = content.decode('utf-8', errors='ignore')
    reader = csv.DictReader(io.StringIO(text))

    for row in reader:
        if any(value == '' or value is None for value in row.values()):
            continue

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

        try:
            response = requests.post(f"{SERVER_URL}/{sensor_id}", data=data)
            print(f"Sent: {row['commodity_name']} - {row['state']} | Status: {response.status_code}")
        except Exception as e:
            print(f"Error sending data: {e}")

        time.sleep(DELAY_SECONDS)  # wait for interval

if __name__ == "__main__":
    main()
    print(">>> CLIENT COMPLETED <<<")
