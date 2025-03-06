# Utiliser une image de base légère
FROM python:3.9

# Définir le répertoire de travail
WORKDIR /app

# Copier le backend
COPY . /app

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port FastAPI
EXPOSE 8000

# Lancer FastAPI
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
