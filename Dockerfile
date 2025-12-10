FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy FastAPI app
COPY api ./api

# # Copy MLflow runs so the model registry works
# COPY mlruns ./mlruns

# Copy exported model and data
COPY chosen_model ./chosen_model
COPY data ./data

# Let MLflow know where the tracking store is
ENV MLFLOW_TRACKING_URI=file:///app/mlruns

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI (note the module path: api/app.py = api.app:app)
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
