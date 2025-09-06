#!/bin/bash

echo " MLOps System Analysis"
echo "========================"

# Check Docker containers
echo " Docker Containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check MLflow models
echo ""
echo " MLflow Analysis:"
curl -s http://localhost:5000/api/2.0/mlflow/registered-models/list | jq '.'

# Check experiments
echo ""
echo " MLflow Experiments:"
curl -s http://localhost:5000/api/2.0/mlflow/experiments/list | jq '.'

# Check Prometheus metrics
echo ""
echo " Prometheus ML Metrics:"
curl -s http://localhost:9090/api/v1/label/__name__/values | grep "ml_" | jq -r '.data[]' | sort

# Check network connectivity
echo ""
echo " Network Connectivity:"
for service in gate-api preprocess-api train-api predict-api mlflow; do
    if docker exec $service curl -s http://localhost:5000 >/dev/null; then
        echo "$service can reach MLflow"
    else
        echo " $service cannot reach MLflow"
    fi
done

# Check data volumes
echo ""
echo " Data Volume Analysis:"
for container in airflow train-api preprocess-api; do
    echo " $container data volume:"
    docker exec $container ls -la /app/data/ 2>/dev/null || echo "No data volume mounted"
done

# Check MLflow artifacts
echo ""
echo " MLflow Artifacts Check:"
MLFLOW_RUNS=$(curl -s http://localhost:5000/api/2.0/mlflow/experiments/list | jq -r '.experiments[].experiment_id')
for run_id in $MLFLOW_RUNS; do
    echo "Experiment $run_id artifacts:"
    curl -s http://localhost:5000/api/2.0/mlflow/artifacts/list?run_id=$run_id | jq '.'
done