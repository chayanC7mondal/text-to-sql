# Deployment Guide

## Prerequisites

- Docker and Docker Compose
- Python 3.10+
- OpenAI API key (optional, for GPT-4)
- Anthropic API key (optional, for Claude)
- At least 8GB RAM for running all services

## Quick Start

### 1. Clone and Setup

```bash
cd text-to-sql
cp .env.example .env
# Edit .env with your API keys
```

### 2. Start Infrastructure

```bash
docker-compose up -d
```

This starts:
- PostgreSQL (port 5432)
- Redis (port 6379)
- MLflow (port 5000)
- Prometheus (port 9090)
- Grafana (port 3000)

### 3. Initialize Database

```bash
# Wait for PostgreSQL to be ready
docker-compose logs -f postgres

# In another terminal, run initialization
docker-compose exec postgres psql -U readonly_user -d ecommerce -f /docker-entrypoint-initdb.d/init.sql
```

### 4. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 5. Start the API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Verify Deployment

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Test query
curl -X POST http://localhost:8000/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"question": "Show me all customers"}'
```

## Development Setup

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app --cov-report=html

# Run specific test file
pytest tests/test_ml_pipeline.py -v
```

### Code Quality

```bash
# Type checking
mypy app/

# Format code (if using black)
black app/

# Lint (if using ruff)
ruff check app/
```

## Production Deployment

### Environment Variables

Set these in production:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@host:5432/db
DATABASE_POOL_SIZE=50
DATABASE_MAX_OVERFLOW=20

# Redis
REDIS_URL=redis://host:6379/0
REDIS_CACHE_TTL=7200

# LLM API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Security
MAX_QUERY_EXECUTION_TIME=60
MAX_QUERY_ROWS_RETURNED=50000

# Observability
MLFLOW_TRACKING_URI=http://mlflow:5000
ENABLE_PROMETHEUS=true
```

### Docker Production Build

```bash
# Build image
docker build -t text-to-sql:latest .

# Run with production config
docker run -d \
  --name text-to-sql \
  -p 8000:8000 \
  --env-file .env.production \
  text-to-sql:latest
```

### Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: text-to-sql
spec:
  replicas: 3
  selector:
    matchLabels:
      app: text-to-sql
  template:
    metadata:
      labels:
        app: text-to-sql
    spec:
      containers:
      - name: api
        image: text-to-sql:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: db-secrets
              key: url
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

### Monitoring

Access dashboards:
- Grafana: http://localhost:3000 (admin/admin)
- Prometheus: http://localhost:9090
- MLflow: http://localhost:5000

## Troubleshooting

### Database Connection Issues

```bash
# Check PostgreSQL status
docker-compose ps postgres

# View logs
docker-compose logs postgres

# Test connection
docker-compose exec postgres psql -U readonly_user -d ecommerce
```

### Redis Connection Issues

```bash
# Check Redis status
docker-compose ps redis

# Test connection
docker-compose exec redis redis-cli ping
```

### Model Loading Issues

```bash
# Check if models directory exists
ls -la models/

# Download pre-trained models if needed
python scripts/download_models.py
```

### Performance Issues

1. Check query performance predictions
2. Review cache hit rate in metrics
3. Check database connection pool utilization
4. Review ML model selection in logs

## Scaling

### Horizontal Scaling

Run multiple API instances behind a load balancer:

```bash
# Scale with Docker Compose
docker-compose up -d --scale api=3
```

### Vertical Scaling

Increase resources in `docker-compose.yml`:

```yaml
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
```

### Database Scaling

- Use connection pooling (PgBouncer)
- Add read replicas
- Partition large tables
- Add appropriate indexes

## Security

### Production Security Checklist

- [ ] Change default Grafana password
- [ ] Use strong database passwords
- [ ] Enable SSL for database connections
- [ ] Use secret management (Vault, AWS Secrets Manager)
- [ ] Enable rate limiting
- [ ] Set up authentication/authorization
- [ ] Enable audit logging
- [ ] Regular security updates

### API Key Management

Never commit API keys. Use:
- Environment variables
- Secret management systems
- Kubernetes secrets
- AWS Parameter Store / Secrets Manager

## Backup and Recovery

### Database Backup

```bash
# Backup
docker-compose exec postgres pg_dump -U readonly_user ecommerce > backup.sql

# Restore
docker-compose exec -T postgres psql -U readonly_user ecommerce < backup.sql
```

### Model Backup

```bash
# Backup models
tar -czf models-backup.tar.gz models/

# Restore
tar -xzf models-backup.tar.gz
```

## Cost Optimization

1. **Caching**: Enable intelligent caching to reduce LLM API calls
2. **Model Selection**: Use local models for simple queries
3. **Connection Pooling**: Optimize database connection pool size
4. **Resource Limits**: Set appropriate CPU/memory limits
5. **Autoscaling**: Scale based on traffic patterns

## Maintenance

### Regular Tasks

- Daily: Review anomaly alerts
- Weekly: Check cache hit rates and adjust thresholds
- Monthly: Retrain ML models with new feedback data
- Quarterly: Review and update security patches

### Model Retraining

```bash
# Retrain performance predictor
python scripts/retrain_models.py --model performance

# Retrain anomaly detector
python scripts/retrain_models.py --model anomaly
```

## Support

For issues:
1. Check logs: `docker-compose logs -f`
2. Review metrics in Grafana
3. Check MLflow experiments
4. Review troubleshooting section above
