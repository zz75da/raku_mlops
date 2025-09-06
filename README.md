# Rakuten MLOps Platform

A complete MLOps platform for multimodal (text + image) classification.

## Architecture

- **Orchestration**: Apache Airflow
- **APIs**: FastAPI microservices (gate-api, preprocess-api, train-api, predict-api)
- **Tracking**: MLflow
- **Storage**: MinIO (S3-compatible)
- **Monitoring**: Prometheus + Grafana
- **Data Versioning**: DVC + DagsHub

## Services

1. **gate-api**: Authentication and user management
2. **preprocess-api**: Feature extraction for text and images
3. **train-api**: Model training with MLflow integration
4. **predict-api**: Model serving and predictions
5. **mlflow**: Experiment tracking and model registry

## Setup

\\\ash
docker-compose up -d
\\\

## Data Management

Large datasets are managed with DVC and stored on DagsHub S3:

\\\ash
# Pull data
dvc pull

# Push data updates
dvc push
\\\

## Repository Structure

\\\
rakuten_mlops_services/
├── airflow/          # Airflow DAGs and configuration
├── gate-api/         # Authentication service
├── preprocess-api/   # Feature extraction
├── train-api/        # Model training
├── predict-api/      # Prediction service
├── monitoring/       # Prometheus & Grafana configs
├── data/            # Datasets (DVC managed)
├── artifacts/       # Model artifacts (DVC managed)
└── docker-compose.yml
\\\

## Repositories

- **Code**: https://github.com/zz75da/raku_mlops
- **Data & Experiments**: https://dagshub.com/zz75da/raku_mlops

## Testing

The project includes a comprehensive test suite with both unit and integration tests.

### Test Structure

- \	ests/unit/\: Unit tests for individual components
  - \	est_preprocess.py\: Tests for preprocessing functions
  - \	est_models.py\: Tests for ML models and utilities
- \	ests/integration/\: Integration tests
  - \	est_api_integration.py\: API integration tests
  - \	est_workflow.py\: End-to-end workflow tests
- \	ests/conftest.py\: Shared test fixtures and configuration
- \	ests/run_tests.py\": Main test runner script

### Running Tests

\\\ash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
python tests/run_tests.py

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v

# Run with coverage report
pytest --cov=preprocess-api --cov=train-api --cov=gate-api --cov=predict-api tests/ --cov-report=html
\\\

### Test Dependencies

Test-specific dependencies are in \equirements-test.txt\ and include:
- pytest
- pytest-cov
- requests-mock
- httpx
