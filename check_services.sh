#!/bin/bash
# Health check script for Docker Compose stack

# Stop on error
set -e

# Services list (container names in docker-compose)
SERVICES=(
  postgres
  minio
  minio-client
  mlflow
  gate-api
  preprocess-api
  train-api
  predict-api
  prometheus
  grafana
  streamlit
  airflow
  airflow-init
  pushgateway
)

echo "Checking containers..."
docker compose ps

echo ""
echo "============================================"
echo " Health checks & last 30 log lines per service"
echo "============================================"

FAILED=0

for svc in "${SERVICES[@]}"; do
  echo ""
  echo "Service: $svc"
  echo "--------------------------------------------"

  STATUS=$(docker inspect --format='{{.State.Status}}' $svc 2>/dev/null || echo "not_found")
  HEALTH=$(docker inspect --format='{{.State.Health.Status}}' $svc 2>/dev/null || echo "no_healthcheck")

  echo "Status      : $STATUS"
  echo "Healthcheck : $HEALTH"

  echo " Logs (last 30 lines):"
  docker compose logs --tail=30 $svc || echo "⚠️ No logs for $svc"

  if [[ "$STATUS" != "running" ]] || [[ "$HEALTH" == "unhealthy" ]] || [[ "$STATUS" == "exited" && "$svc" != "airflow-init" ]]; then
    echo "❌ Service $svc is not healthy (status=$STATUS, health=$HEALTH)"
    FAILED=1
  else
    echo "✅ Service $svc looks fine."
  fi

  echo "--------------------------------------------"
done

echo ""
echo "============================================"
echo "HTTP endpoint checks"
echo "============================================"

check_http() {
  local url=$1
  local name=$2
  if curl -fsS "$url" > /dev/null; then
    echo "✅ $name reachable at $url"
  else
    echo "❌ $name NOT reachable at $url"
    FAILED=1
  fi
}

check_http "http://localhost:5000" "MLflow"
check_http "http://localhost:8080/health" "Airflow Webserver"
check_http "http://localhost:5004" "Gate API"
check_http "http://localhost:5001" "Preprocess API"
check_http "http://localhost:5002" "Train API"
check_http "http://localhost:5003" "Predict API"
check_http "http://localhost:9090" "Prometheus"
check_http "http://localhost:3000" "Grafana"
check_http "http://localhost:8501" "Streamlit"
check_http "http://localhost:9091" "Prometheus Pushgateway"

echo ""
if [[ $FAILED -ne 0 ]]; then
  echo "❌ Some services failed. Check logs above."
  exit 1
else
  echo "🚀 All services running and healthy!"
fi
