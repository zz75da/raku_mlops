#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE} Starting Comprehensive MLOps Microservices Test...${NC}"
echo "=========================================================="

# Get JWT Token
echo -e "${YELLOW} Getting JWT Token...${NC}"
TOKEN_RESPONSE=$(curl -s -X POST http://localhost:5004/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin_pass"}')

if echo "$TOKEN_RESPONSE" | grep -q "token"; then
    TOKEN=$(echo "$TOKEN_RESPONSE" | grep -o '"token":"[^"]*' | cut -d'"' -f4)
    echo -e "${GREEN}Token obtained: ${TOKEN:0:20}...${NC}"
else
    echo -e "${RED} Failed to get token${NC}"
    echo "Response: $TOKEN_RESPONSE"
    exit 1
fi

HEADERS=(-H "Content-Type: application/json" -H "Authorization: Bearer $TOKEN")

# Test function
test_endpoint() {
    local name=$1
    local url=$2
    local method=$3
    local data=$4
    
    echo -e "${YELLOW}Testing $name...${NC}"
    echo "URL: $url"
    
    if [ "$method" = "POST" ] && [ -n "$data" ]; then
        RESPONSE=$(curl -s -X $method "$url" "${HEADERS[@]}" -d "$data")
    else
        RESPONSE=$(curl -s -X $method "$url" "${HEADERS[@]}")
    fi
    
    if [ $? -eq 0 ] && [ -n "$RESPONSE" ]; then
        echo -e "${GREEN} Success! Response:${NC}"
        echo "$RESPONSE" | jq . 2>/dev/null || echo "$RESPONSE"
        echo "----------------------------------------"
        return 0
    else
        echo -e "${RED} Failed!${NC}"
        echo "Response: $RESPONSE"
        echo "----------------------------------------"
        return 1
    fi
}

# Health Checks
echo -e "${BLUE} Health Checks${NC}"
test_endpoint "Gate API Health" "http://localhost:5004/health" "GET"
test_endpoint "Preprocess API Health" "http://localhost:5001/health" "GET"
test_endpoint "Train API Health" "http://localhost:5002/health" "GET"
test_endpoint "Predict API Health" "http://localhost:5003/health" "GET"
test_endpoint "MLflow Health" "http://localhost:5000" "GET"

# MLflow Check
echo -e "${BLUE} MLflow Status Check${NC}"
MLFLOW_STATUS=$(curl -s http://localhost:5000/api/2.0/mlflow/experiments/list)
if echo "$MLFLOW_STATUS" | grep -q "experiments"; then
    echo -e "${GREEN} MLflow is running${NC}"
    echo "Experiments found"
else
    echo -e "${RED} MLflow not responding properly${NC}"
fi

# Test Preprocess API
echo -e "${BLUE} Testing Preprocess API${NC}"
test_endpoint "Text Preprocessing" "http://localhost:5001/extract-features" "POST" '[
    {"description": "Amazing wireless headphones with noise cancellation"},
    {"description": "Professional gaming keyboard with RGB lighting"}
]'

# Test Training
echo -e "${BLUE} Testing Training API${NC}"
test_endpoint "Start Training" "http://localhost:5002/train" "POST" '{"test_size": 0.2, "epochs": 2}'

# Wait for training to complete
echo -e "${YELLOW} Waiting 30 seconds for training to complete...${NC}"
sleep 30

# Check MLflow for trained model
echo -e "${BLUE} Checking MLflow for Trained Model${NC}"
MLFLOW_MODELS=$(curl -s http://localhost:5000/api/2.0/mlflow/registered-models/list)
if echo "$MLFLOW_MODELS" | grep -q "neural_network_model"; then
    echo -e "${GREEN} Model registered in MLflow${NC}"
else
    echo -e "${RED} Model not found in MLflow${NC}"
    echo "MLflow response: $MLFLOW_MODELS"
fi

# Test Prediction
echo -e "${BLUE} Testing Prediction API${NC}"
test_endpoint "Text Prediction" "http://localhost:5003/predict" "POST" '{
    "description": "High quality wireless headphones with premium sound"
}'

test_endpoint "Empty Prediction" "http://localhost:5003/predict" "POST" '{
    "description": null,
    "image_features": null
}'

# Test Admin endpoints
echo -e "${BLUE} Testing Admin Endpoints${NC}"
test_endpoint "Model Reload" "http://localhost:5003/reload-model" "POST"

# System Metrics Check
echo -e "${BLUE} System Metrics Check${NC}"
PROMETHEUS_METRICS=$(curl -s http://localhost:9090/api/v1/label/__name__/values)
if echo "$PROMETHEUS_METRICS" | grep -q "ml_"; then
    echo -e "${GREEN} Prometheus metrics found:${NC}"
    echo "$PROMETHEUS_METRICS" | grep "ml_"
else
    echo -e "${RED} No ML metrics in Prometheus${NC}"
fi

# Grafana Check
echo -e "${BLUE} Grafana Check${NC}"
GRAFANA_HEALTH=$(curl -s http://localhost:3000/api/health)
if echo "$GRAFANA_HEALTH" | grep -q "database"; then
    echo -e "${GREEN} Grafana is running${NC}"
else
    echo -e "${RED} Grafana not responding${NC}"
fi

# Final Summary
echo -e "${BLUE}==========================================================${NC}"
echo -e "${BLUE} TEST SUMMARY${NC}"
echo -e "${BLUE}==========================================================${NC}"

# Check all critical services
SERVICES=(
    "gate-api:5004"
    "preprocess-api:5001" 
    "train-api:5002"
    "predict-api:5003"
    "mlflow:5000"
    "prometheus:9090"
    "grafana:3000"
)

for service in "${SERVICES[@]}"; do
    if curl -s "http://${service}" >/dev/null; then
        echo -e "${GREEN} $service is UP${NC}"
    else
        echo -e "${RED} $service is DOWN${NC}"
    fi
done

echo -e "${BLUE}==========================================================${NC}"
echo -e "${GREEN}🎉 Test completed! Check above for any failures.${NC}"
echo -e "${BLUE}==========================================================${NC}"