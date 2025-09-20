# Use official Airflow image as base (BETTER than python slim)
FROM apache/airflow:2.10.2

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    postgresql-client \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Create log directory with proper permissions to avoid Windows issues
RUN mkdir -p /tmp/airflow/logs && chmod -R 777 /tmp/airflow/logs

USER airflow

# Copy and install requirements
COPY airflow/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install HTTP provider for Airflow (CRITICAL)
RUN pip install apache-airflow-providers-http

RUN pip install PyJWT

RUN pip install mlflow prometheus-client

# Create directories for data access
RUN mkdir -p /opt/airflow/data /opt/airflow/artifacts

WORKDIR /opt/airflow

# Copy DAGs and configuration
COPY airflow/dags /opt/airflow/dags
COPY airflow/plugins /opt/airflow/plugins
COPY airflow/airflow.cfg /opt/airflow/airflow.cfg
COPY params.yaml /app/params.yaml

# Expose ports
EXPOSE 8080

# Default command (will be overridden by docker-compose)
CMD ["bash"]