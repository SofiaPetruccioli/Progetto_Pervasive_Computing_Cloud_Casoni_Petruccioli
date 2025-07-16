# Immagine base con Python
FROM python:3.11-slim

# Imposta la working directory
WORKDIR /app

# Copia tutti i file del progetto
COPY . .
COPY secret.py .
# Installa le dipendenze
RUN pip install --no-cache-dir -r requirements.txt

# Espone la porta usata da Cloud Run
EXPOSE 8080

# Comando per eseguire l'app Flask con gunicorn
CMD ["sh", "-c", "exec gunicorn -b :$PORT main:app"]


