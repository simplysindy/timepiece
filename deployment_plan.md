# Final Cloud Deployment Plan for Watch Forecasting (GCP Version)

## 📋 Implementation Checklist & Progress Tracking

> **Instructions for Updating This Checklist:**
> 1. Check off tasks as you complete them: `- [x] Task completed`
> 2. When a phase is complete, update the Developer Notes section with what was accomplished
> 3. Add any issues encountered and their solutions
> 4. Note any changes to the plan or upcoming phase requirements
> 5. Update the "Next Phase Prep" section with specific setup needed for the following phase

### Phase 1: Local API Development (Week 1-2)
**Status:** ✅ Complete

**Tasks:**
- [x] Create `src/api/` directory structure
- [x] Extract prediction logic into `ForecastService` class
- [x] Build basic FastAPI app with `/predict` and `/health` endpoints
- [x] Test API locally: `uvicorn src.api.main:app --reload`
- [ ] Add gRPC service (optional - can skip initially)
- [x] Update this checklist with completion notes

**Developer Notes - Phase 1:**
```
Key Accomplishments:
✅ Created complete API architecture with src/api/ directory structure
✅ Developed ForecastService class that encapsulates all prediction logic
✅ Built comprehensive FastAPI application with 6 endpoints:
   - GET /health - Service health check with model/data status
   - GET /watches - List all available watch IDs (5 Patek Philippe models)
   - GET /models - List all available ML models (lightgbm, ridge, xgboost, etc.)
   - POST /predict - Single watch prediction with configurable horizon/model
   - POST /predict/batch - Batch predictions for multiple watches
   - GET / - API info and documentation links
✅ Successfully tested all endpoints locally with uvicorn
✅ Verified predictions working with multiple models (lightgbm, ridge)
✅ API properly handles 5 available watch models and 5 ML algorithms

Next Phase Prep:
- Install Google Cloud SDK: `gcloud` CLI tools
- Set up GCP project and enable required APIs
- Ensure pip dependencies include: fastapi, uvicorn, google-cloud-storage
- Create requirements.txt for Cloud Functions deployment
- Upload existing models to Cloud Storage bucket

Additional notes:
- Data path fixed to use correct location: data/output/featured_data.csv
- All endpoints return proper JSON responses with error handling
- Service initializes once and caches models for performance
- Ready for GCP deployment with minimal code changes needed
```

### Phase 2: GCP Setup (Week 2-3)
**Status:** ✅ Complete

**Tasks:**
- [x] Set up GCP account and install gcloud CLI
- [x] Create GCP project: `timepiece-473511` (project already existed)
- [x] Enable required APIs: `gcloud services enable cloudfunctions.googleapis.com storage.googleapis.com cloudbuild.googleapis.com run.googleapis.com`
- [x] Create Cloud Storage bucket for models: `gsutil mb gs://timepiece-watch-models`
- [x] Upload existing models: `gsutil -m cp -r data/output/models/* gs://timepiece-watch-models/models/`
- [x] Create `main.py` and `requirements.txt` for Cloud Functions
- [ ] Deploy first function: `gcloud functions deploy predict --runtime python311`
- [x] Update this checklist with completion notes

**Developer Notes - Phase 2:**
```
Key Accomplishments:
✅ Google Cloud CLI installed and configured successfully
✅ Authenticated with GCP account (simplysindyhua@gmail.com)
✅ Connected to existing timepiece-473511 project
✅ Enabled all required APIs:
   - Cloud Functions API (cloudfunctions.googleapis.com)
   - Cloud Storage API (storage.googleapis.com)
   - Cloud Build API (cloudbuild.googleapis.com)
   - Cloud Run API (run.googleapis.com)
✅ Created Cloud Storage bucket: gs://timepiece-watch-models
✅ Successfully uploaded all 42 model files (92.0 MiB total)
✅ Uploaded featured_data.csv to Cloud Storage
✅ Updated requirements.txt with FastAPI and GCP dependencies
✅ Infrastructure ready for Cloud Functions/Cloud Run deployment

Next Phase Prep:
- Create main.py for Cloud Functions entry point
- Test Cloud Functions deployment with a simple function
- Modify ForecastService to load models from Cloud Storage
- Deploy to Cloud Run as alternative to Cloud Functions
- Update Streamlit app to use cloud API endpoints

Additional notes:
- Bucket structure: gs://timepiece-watch-models/models/ and gs://timepiece-watch-models/data/
- All model types uploaded: lightgbm, xgboost, ridge, linear, random_forest
- Both horizon-specific models (__h1 to __h7) and general models available
- Ready to proceed with actual deployment to cloud services
```

### Phase 3: Streamlit Integration (Week 3-4)
**Status:** 🔄 In Progress

**Tasks:**
- [x] Add environment variable for Cloud Function URL in Streamlit app
- [x] Create toggle between local and cloud inference
- [x] Add developer API documentation section
- [ ] Test end-to-end functionality with Cloud Functions
- [ ] Deploy Streamlit app to Cloud Run (optional)
- [ ] Create .proto file for gRPC (if implemented)
- [x] Update this checklist with completion notes

**Developer Notes - Phase 3:**
```
Key Accomplishments:
✅ Streamlit now reads `GCP_API_URL`/`WATCH_API_URL` (or sidebar input) to call the deployed prediction API.
✅ Added a sidebar backend toggle with cached helpers for `/watches`, `/models`, and `/predict`, preserving the original local flow.
✅ Inserted an "API for developers" expander plus a raw-response viewer to simplify debugging.
✅ Documented round-trip validation commands in docs/cloud_round_trip.md and verified `streamlit_app.py` compiles.

Next Phase Prep:
- Exercise a full Streamlit → Cloud Run round trip and capture screenshots for docs.
- Decide on the optional gRPC surface and `.proto` scaffold.
- Package Streamlit for Cloud Run deployment once remote endpoints stabilize.

Additional notes:
- Cloud mode reuses local feature data for charting while live predictions come from the remote API.
- Remote calls use `st.cache_data` and surface clear errors if the service is offline.
- Local inference behaviour (Hydra defaults, on-disk models) remains unchanged when the toggle is set to "Local models".
```

### Phase 4: Production Ready (Week 4+)
**Status:** ⏳ Not Started | 🔄 In Progress | ✅ Complete

**Tasks:**
- [ ] Set up Cloud Build for auto-deployment from GitHub
- [ ] Add error handling and Cloud Logging
- [ ] Configure custom domain with Cloud Endpoints (optional)
- [ ] Set up monitoring with Cloud Monitoring
- [ ] Document API endpoints and usage
- [ ] Configure alerts for errors and latency
- [ ] Update this checklist with completion notes

**Developer Notes - Phase 4:**
```
Key Accomplishments:
_This section will be populated by the developer_

Next Phase Prep:
_This section will be populated by the developer_

Additional notes:
_This section will be populated by the developer_
```

## Overview

Practical, phased approach to transform your local ML pipeline into a cloud-accessible API service using Google Cloud Platform while keeping the existing Streamlit interface and adding developer-friendly access.

## 🎯 Current Architecture Understanding

### Existing Pipeline Structure
Based on the current CLAUDE.md specifications:

```
src/
├── scraper/          # Data collection from WatchCharts.com
├── data_prep/        # Feature engineering (80+ features)
├── training/         # Multi-model training (Ridge, LightGBM, XGBoost, etc.)
└── inference/        # Prediction generation + Streamlit UI
```

**Key Commands:**
- `python -m src.scraper.scraper` - Web scraping pipeline
- `python -m src.data_prep.data_prep` - Feature engineering
- `python -m src.training.training` - Model training
- `python -m src.inference.inference` - Batch predictions
- `streamlit run src/inference/streamlit_app.py` - Interactive frontend

**Configuration:** Hydra-based with `conf/*.yaml` files for all pipelines

## 🚀 Implementation Phases (GCP-Focused Approach)

### Phase 1: Core API Development
Start with the foundation - making your inference logic accessible via APIs.

#### 1.1 Extract & Modularize Inference Logic
```python
# src/api/forecast_service.py
class ForecastService:
    def __init__(self):
        self.models = self._load_models()
        self.scalers = self._load_scalers()

    def predict(self, watch_id: str, horizon: int, model_type: str):
        # Core prediction logic extracted from current inference.py
        pass
```

#### 1.2 Build REST API with FastAPI
```python
# src/api/main.py
from fastapi import FastAPI
from .forecast_service import ForecastService

app = FastAPI(title="Watch Price Forecasting API")
service = ForecastService()

@app.post("/v1/predict")
async def predict(watch_id: str, horizon: int = 7):
    return service.predict(watch_id, horizon, "lightgbm")

@app.get("/v1/health")
async def health():
    return {"status": "healthy"}
```

#### 1.3 Add gRPC Service (Parallel to REST)
```python
# src/api/grpc_server.py
import grpc
from concurrent import futures
from . import forecast_pb2, forecast_pb2_grpc

class ForecastServicer(forecast_pb2_grpc.ForecastServicer):
    def Predict(self, request, context):
        # Share the same ForecastService instance
        return forecast_pb2.PredictResponse(...)

# Run on port 50051 alongside FastAPI on 8000
```

### Phase 2: GCP Cloud Deployment
Using Google Cloud Functions for serverless deployment and Cloud Storage for model hosting.

#### 2.1 Package for Cloud Functions
```python
# main.py (Cloud Functions entry point)
import functions_framework
from google.cloud import storage
from src.api.forecast_service import ForecastService

# Initialize service globally for reuse across requests
service = None

def initialize_service():
    """Load models from Cloud Storage on cold start"""
    global service
    if service is None:
        service = ForecastService()
    return service

@functions_framework.http
def predict(request):
    """HTTP Cloud Function for predictions"""
    service = initialize_service()
    
    request_json = request.get_json(silent=True)
    watch_id = request_json.get('watch_id')
    horizon = request_json.get('horizon', 7)
    
    result = service.predict(watch_id, horizon, "lightgbm")
    
    return {'prediction': result, 'status': 'success'}

# For FastAPI on Cloud Run (alternative to Cloud Functions)
from src.api.main import app
# Deploy this directly to Cloud Run
```

#### 2.2 Model Storage Setup
```bash
# Create Cloud Storage bucket
gsutil mb -p watch-forecast-project gs://watch-forecast-models

# Upload models to Cloud Storage
gsutil -m cp -r data/output/models/* gs://watch-forecast-models/

# Set public read access (if needed)
gsutil iam ch allUsers:objectViewer gs://watch-forecast-models
```

```python
# Model loading from Cloud Storage
from google.cloud import storage

def load_models_from_gcs(bucket_name="watch-forecast-models"):
    """Download models from Cloud Storage"""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Download to /tmp/ in Cloud Functions
    for blob in bucket.list_blobs(prefix="models/"):
        local_path = f"/tmp/{blob.name}"
        blob.download_to_filename(local_path)
    
    return load_local_models("/tmp/models/")
```

#### 2.3 Deployment Configuration
```yaml
# cloudbuild.yaml (for Cloud Build CI/CD)
steps:
  # Install dependencies
  - name: 'python:3.11'
    entrypoint: pip
    args: ['install', '-r', 'requirements.txt', '-t', '.']
  
  # Deploy to Cloud Functions
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    args:
      - gcloud
      - functions
      - deploy
      - predict
      - --runtime=python311
      - --trigger-http
      - --allow-unauthenticated
      - --memory=2GB
      - --timeout=300
      - --set-env-vars=MODEL_BUCKET=watch-forecast-models

# requirements.txt for Cloud Functions
fastapi==0.104.1
google-cloud-storage==2.10.0
pandas==2.1.3
scikit-learn==1.3.2
lightgbm==4.1.0
xgboost==2.0.2
```

#### 2.4 Cloud Run Alternative (for FastAPI)
```dockerfile
# Dockerfile for Cloud Run
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ ./src/
COPY conf/ ./conf/

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]
```

```bash
# Deploy to Cloud Run
gcloud run deploy watch-forecast-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

### Phase 3: Frontend Integration

#### 3.1 Update Streamlit to Use GCP API
```python
# src/inference/streamlit_app.py
import streamlit as st
import requests
import os

# Cloud Function URL or Cloud Run URL
API_URL = os.getenv("GCP_API_URL", "http://localhost:8000")

if st.button("Predict"):
    if USE_CLOUD_API:
        # For Cloud Functions
        cf_url = "https://predict-abc123-uc.a.run.app"
        response = requests.post(cf_url, json={
            "watch_id": selected_watch,
            "horizon": horizon_days
        })
    else:
        # Existing local inference code
```

#### 3.2 Deploy Streamlit to Cloud Run
```yaml
# streamlit.dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "src/inference/streamlit_app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
# Deploy Streamlit app
gcloud run deploy watch-forecast-ui \
  --source . \
  --platform managed \
  --port 8501 \
  --allow-unauthenticated
```

#### 3.3 Add Developer Section
```python
with st.expander("API for Developers"):
    st.code(f"""
    # REST API (Cloud Functions)
    curl -X POST https://us-central1-watch-forecast.cloudfunctions.net/predict \\
      -H "Content-Type: application/json" \\
      -d '{{"watch_id": "rolex-submariner", "horizon": 7}}'

    # Python Client
    import requests
    response = requests.post(
        "https://watch-forecast-api-abc123.a.run.app/v1/predict",
        json={{"watch_id": "rolex-submariner", "horizon": 7}}
    )

    # gRPC (for high-performance needs)
    # Deploy to Cloud Run with gRPC enabled
    channel = grpc.insecure_channel('watch-forecast-grpc.a.run.app:443')
    """)
```

### Phase 4: CI/CD Setup

#### 4.1 Cloud Build Triggers
```yaml
# cloudbuild.yaml
steps:
  # Run tests
  - name: 'python:3.11'
    entrypoint: 'python'
    args: ['-m', 'pytest', 'tests/']
  
  # Deploy Cloud Function
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    args:
      - gcloud
      - functions
      - deploy
      - predict
      - --source=.
      - --trigger-http
      - --runtime=python311
      - --region=us-central1
  
  # Deploy Cloud Run (if using FastAPI)
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-t', 'gcr.io/$PROJECT_ID/watch-forecast-api', '.']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/watch-forecast-api']
  
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    args:
      - gcloud
      - run
      - deploy
      - watch-forecast-api
      - --image=gcr.io/$PROJECT_ID/watch-forecast-api
      - --platform=managed
      - --region=us-central1

# Trigger on push to main branch
options:
  logging: CLOUD_LOGGING_ONLY
```

#### 4.2 GitHub Actions Alternative
```yaml
# .github/workflows/deploy-gcp.yml
name: Deploy to GCP
on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - id: 'auth'
        uses: 'google-github-actions/auth@v1'
        with:
          credentials_json: '${{ secrets.GCP_SA_KEY }}'
      
      - name: 'Set up Cloud SDK'
        uses: 'google-github-actions/setup-gcloud@v1'
      
      - name: 'Deploy Cloud Function'
        run: |
          gcloud functions deploy predict \
            --runtime python311 \
            --trigger-http \
            --allow-unauthenticated \
            --memory 2GB \
            --timeout 300 \
            --set-env-vars MODEL_BUCKET=watch-forecast-models
```

## ❌ What to SKIP (Not Worth Time Now)

**Don't Over-Engineer Initially:**
- Complex orchestration (Cloud Composer/Airflow) - use Cloud Scheduler for simple jobs
- Multiple cloud providers - stick with GCP only
- Firestore/BigQuery (keep using Cloud Storage for now) - simpler for model files
- Identity Platform authentication (add later if needed) - start with open API
- GKE/Kubernetes (overkill for your scale) - Cloud Run/Functions are simpler
- Extensive monitoring (just use Cloud Logging basics) - avoid analysis paralysis

## 🔧 Technical Implementation Details

### Key Integration Points with Existing Code

#### Leverage Current `inference.py` Structure
```python
# Reuse existing logic from src/inference/inference.py:generate_predictions()
from src.inference.inference import generate_predictions
from src.training.features import prepare_features

class ForecastService:
    def __init__(self):
        # Load models from GCS on initialization
        self.models = self._load_models_from_gcs()
    
    def _load_models_from_gcs(self):
        """Load models from Cloud Storage"""
        from google.cloud import storage
        client = storage.Client()
        bucket = client.bucket("watch-forecast-models")
        # Download and cache models
        return loaded_models
    
    def predict_single(self, watch_id: str, horizon: int, model_name: str):
        # Use existing generate_predictions function
        return generate_predictions(
            data_path="data/output/featured_data.csv",
            model_dir="/tmp/models",  # Models downloaded from GCS
            models=[model_name],
            horizons=[horizon],
            asset_column="watch_id"
        )
```

#### Preserve Hydra Configuration
```python
# Keep using existing conf/*.yaml structure
from omegaconf import OmegaConf

# In Cloud Functions
def initialize_config():
    """Load config from Cloud Storage or include in deployment"""
    from google.cloud import storage
    client = storage.Client()
    bucket = client.bucket("watch-forecast-config")
    blob = bucket.blob("conf/inference.yaml")
    config_str = blob.download_as_text()
    return OmegaConf.create(yaml.safe_load(config_str))
```

### Model Management Strategy

#### Cloud Storage Organization
```
gs://watch-forecast-models/
├── models/
│   ├── lightgbm/
│   │   ├── model_7d.pkl
│   │   └── model_30d.pkl
│   ├── xgboost/
│   └── ridge/
├── scalers/
│   └── standard_scaler.pkl
└── metadata/
    └── model_versions.json
```

### Simple Monitoring

#### Cloud Logging Integration
```python
import google.cloud.logging
import logging

# Setup Cloud Logging
client = google.cloud.logging.Client()
client.setup_logging()

logger = logging.getLogger(__name__)

@functions_framework.http
def predict(request):
    start_time = time.time()
    try:
        result = service.predict(...)
        logger.info(f"Prediction successful: {watch_id}, latency: {time.time() - start_time:.2f}s")
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {watch_id}", exc_info=True)
        return {"error": str(e)}, 500
```

#### Cloud Monitoring Alerts
```yaml
# monitoring.yaml - Deploy with gcloud
alertPolicy:
  displayName: "High Latency Alert"
  conditions:
    - displayName: "Cloud Function latency > 5s"
      conditionThreshold:
        filter: |
          resource.type="cloud_function"
          resource.labels.function_name="predict"
          metric.type="cloudfunctions.googleapis.com/function/execution_times"
        comparison: COMPARISON_GT
        thresholdValue: 5000  # milliseconds
```

## 🎯 Success Metrics

### Technical Goals
- **API Response Time**: < 2 seconds for single predictions (Cloud Functions)
- **Availability**: 99.9% uptime with GCP SLA
- **Cost**: < $50/month for moderate usage (1000 predictions/day)
- **Deployment Time**: < 5 minutes via Cloud Build

### Business Goals
- **Developer Adoption**: Provide clear API documentation and examples
- **User Experience**: Maintain existing Streamlit functionality while adding cloud capabilities
- **Scalability**: Auto-scaling with Cloud Functions/Cloud Run
- **Maintainability**: Single codebase supporting both local and cloud deployment

## 💡 Quick Start Commands

```bash
# 1. GCP Setup
gcloud auth login
gcloud config set project watch-forecast-project
gcloud services enable cloudfunctions.googleapis.com storage.googleapis.com

# 2. Local API Development
mkdir src/api
pip install fastapi uvicorn google-cloud-storage
uvicorn src.api.main:app --reload

# 3. Model Upload to Cloud Storage
gsutil mb gs://watch-forecast-models
gsutil -m cp -r data/output/models/* gs://watch-forecast-models/

# 4. Deploy Cloud Function
gcloud functions deploy predict \
  --runtime python311 \
  --trigger-http \
  --allow-unauthenticated \
  --memory 2GB \
  --source . \
  --entry-point predict

# 5. Test Deployment
curl -X POST https://us-central1-watch-forecast.cloudfunctions.net/predict \
  -H "Content-Type: application/json" \
  -d '{"watch_id": "rolex-submariner", "horizon": 7}'

# 6. Deploy to Cloud Run (Alternative)
gcloud run deploy watch-forecast-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

# 7. View Logs
gcloud functions logs read predict --limit 50
gcloud logging read "resource.type=cloud_function" --limit 10
```

## 🔧 Maintenance & Operations

### Regular Tasks
- **Weekly**: Check Cloud Logging for errors and performance metrics
- **Monthly**: Review GCP billing and optimize unused resources
- **Quarterly**: Update model artifacts in Cloud Storage
- **As Needed**: Adjust Cloud Function memory/timeout based on usage

### Cost Optimization Tips
- Use Cloud Functions for sporadic traffic (pay per invocation)
- Use Cloud Run for consistent traffic (better for sustained load)
- Set up lifecycle policies on Cloud Storage to archive old models
- Use Cloud Scheduler to trigger retraining jobs instead of always-on compute

This GCP-focused plan leverages Google Cloud's serverless offerings to deploy your watch forecasting project with minimal complexity while preserving all existing functionality and adding developer-friendly API access.
