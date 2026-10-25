# System Architecture

## High-Level Overview

The Text-to-SQL system is designed as a production-grade ML-powered application with multiple layers of intelligence:

```
User Request → API Gateway → ML Pipeline → SQL Generation → Database → Result
                     ↓              ↓              ↓
                Ambiguity      Ensemble       Safety
                Detection      Selection      Validation
                     ↓              ↓              ↓
                Cache          Performance    Anomaly
                Check          Prediction     Detection
```

## Component Architecture

### 1. API Gateway (FastAPI)
- **Purpose**: Request ingestion, validation, routing
- **Technologies**: FastAPI, Pydantic v2, async/await
- **Features**:
  - Strict payload validation
  - Request lifecycle management
  - Rate limiting
  - Request tracing
  - Error handling

### 2. ML Pipeline
- **Purpose**: Core ML intelligence layer
- **Components**:
  - **Embedding Generator**: sentence-transformers for semantic representation
  - **Ambiguity Detector**: Similarity-based detection with clustering
  - **Performance Predictor**: XGBoost model for execution time prediction
  - **Anomaly Detector**: Isolation Forest for unusual patterns

### 3. SQL Generation Engine
- **Purpose**: Convert natural language to SQL
- **Components**:
  - **Ensemble Manager**: Weighted ensemble of multiple LLMs
  - **Model Selector**: Complexity-based dynamic routing
  - **LLM Providers**: OpenAI, Anthropic, local models
  - **Schema Builder**: Dynamic schema context construction

### 4. Safety Layer
- **Purpose**: Ensure safe query execution
- **Components**:
  - **AST Validator**: sqlglot-based parsing for SELECT-only enforcement
  - **Injection Detector**: Pattern-based SQL injection detection
  - **Read-Only Enforcement**: Database-level read-only credentials

### 5. Cache Layer
- **Purpose**: Intelligent result caching
- **Components**:
  - **Redis Backend**: High-performance caching
  - **Similarity Search**: Embedding-based cache lookup
  - **Eviction Policy**: ML-driven based on access patterns
  - **Warm-up Engine**: Pre-cache common queries

### 6. Observability Layer
- **Purpose**: Monitor and track system performance
- **Components**:
  - **Prometheus Metrics**: Real-time metrics collection
  - **MLflow Tracking**: Experiment and model tracking
  - **Grafana Dashboards**: Visualization
  - **Structured Logging**: JSON-formatted logs

### 7. Database Layer
- **Purpose**: Query execution and data storage
- **Components**:
  - **PostgreSQL**: Primary database with 1.55M+ records
  - **Connection Pool**: AsyncPG connection pooling
  - **Read-Only Replica**: Safe query execution
  - **Schema Manager**: Dynamic schema information

## Data Flow

### Query Processing Flow

```
1. User submits question
   ↓
2. API validates request (Pydantic)
   ↓
3. Check cache (embedding similarity)
   ↓
4. Detect ambiguity (ML pipeline)
   ├─→ If ambiguous: Return clarification options
   └─→ If clear: Continue
   ↓
5. Generate SQL (ensemble selection)
   ├─→ Simple queries: Local model
   ├─→ Medium complexity: Anthropic
   └─→ High complexity: OpenAI GPT-4
   ↓
6. Validate SQL (AST + pattern matching)
   ├─→ If unsafe: Reject with error
   └─→ If safe: Continue
   ↓
7. Predict performance (XGBoost)
   ├─→ If too slow: Warn user
   └─→ If acceptable: Continue
   ↓
8. Detect anomalies (Isolation Forest)
   ├─→ If anomalous: Log alert
   └─→ If normal: Continue
   ↓
9. Execute query (PostgreSQL read-only)
   ↓
10. Format results (natural language summary)
    ↓
11. Cache result (with embedding)
    ↓
12. Return response to user
```

### Feedback Flow

```
1. User provides feedback
   ↓
2. Record in feedback loop
   ↓
3. Update model weights (reinforcement learning)
   ↓
4. Add to training data
   ↓
5. Check retraining trigger (every 100 feedback)
   ↓
6. If triggered: Prepare training data
   ↓
7. Export for fine-tuning
   ↓
8. Retrain models (offline)
   ↓
9. Deploy new model version
```

### A/B Testing Flow

```
1. Create experiment (control vs treatment)
   ↓
2. Assign user to variant (hash-based or random)
   ↓
3. Execute query with assigned model
   ↓
4. Record result (success/failure)
   ↓
5. Update bandit state (Beta distribution)
   ↓
6. Check statistical significance (Fisher's exact)
   ↓
7. If significant: Declare winner
   ↓
8. Roll out winning model
```

## Technology Stack

### Backend Framework
- **FastAPI**: Modern async web framework
- **Pydantic v2**: Data validation and settings
- **Uvicorn**: ASGI server

### Database
- **PostgreSQL 15**: Primary database
- **AsyncPG**: Async PostgreSQL driver
- **SQLAlchemy**: ORM (optional)
- **sqlglot**: SQL parsing and validation

### ML/LLM
- **OpenAI GPT-4**: High-quality SQL generation
- **Anthropic Claude**: Alternative LLM provider
- **sentence-transformers**: Embedding generation
- **XGBoost/LightGBM**: Performance prediction
- **scikit-learn**: ML utilities
- **Isolation Forest**: Anomaly detection

### Caching
- **Redis**: High-performance caching
- **hiredis**: Redis C extension

### Observability
- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **MLflow**: ML experiment tracking
- **Structured logging**: JSON logs

### Testing
- **pytest**: Test framework
- **pytest-asyncio**: Async test support
- **pytest-cov**: Coverage reporting
- **hypothesis**: Property-based testing

### Infrastructure
- **Docker**: Containerization
- **Docker Compose**: Multi-container orchestration
- **Kubernetes**: Production deployment (optional)

## Deployment Architecture

### Development
```
Local Machine
├── FastAPI (uvicorn --reload)
├── PostgreSQL (Docker)
├── Redis (Docker)
└── MLflow (Docker)
```

### Production
```
Load Balancer
    ↓
[API Instance 1] [API Instance 2] [API Instance 3]
    ↓                    ↓                    ↓
        PostgreSQL (Primary + Read Replicas)
                    ↓
                Redis Cluster
                    ↓
                MLflow Server
                    ↓
            Prometheus + Grafana
```

## Security Architecture

### Multi-Layer Security

1. **Application Layer**
   - Pydantic validation
   - Rate limiting
   - Input sanitization

2. **ML Layer**
   - Anomaly detection
   - Pattern matching
   - Confidence thresholds

3. **SQL Layer**
   - AST validation
   - Read-only enforcement
   - SQL injection detection

4. **Database Layer**
   - Read-only user role
   - Connection limits
   - Query timeouts

5. **Infrastructure Layer**
   - Network isolation
   - Secret management
   - Audit logging

## Scalability Considerations

### Horizontal Scaling
- Stateless API design
- Redis for shared state
- Database connection pooling
- Load balancer compatibility

### Vertical Scaling
- Configurable connection pools
- Adjustable cache sizes
- Tunable ML model parameters
- Resource limits (Docker/K8s)

### Performance Optimization
- Intelligent caching (92% hit rate)
- Query prediction (prevent long queries)
- Model selection (right tool for the job)
- Async I/O throughout

## Monitoring & Alerting

### Key Metrics
- Query latency (p50, p95, p99)
- Cache hit rate
- Model accuracy
- Error rate
- Anomaly count
- Active sessions

### Alerts
- High error rate (>5%)
- Low cache hit rate (<70%)
- Model accuracy drop (>10%)
- Anomaly spike (>10/hour)
- Database connection issues

## Data Retention

### Logs
- Application logs: 30 days
- Access logs: 90 days
- Error logs: 180 days

### Metrics
- Prometheus metrics: 15 days resolution
- MLflow experiments: 180 days

### Feedback
- User feedback: 365 days
- Training data: 365 days

### Models
- Model versions: Keep last 10
- Model artifacts: 365 days
