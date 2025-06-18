# Phase 6: CI/CD Pipeline Documentation
# Complete Automation of Testing, Deployment, and Retraining Workflows

## Overview

This CI/CD pipeline provides comprehensive automation for the Dynamic Pricing ML system including:

1. **Continuous Integration**: Code quality, testing, and validation
2. **Continuous Deployment**: Automated deployment to staging and production
3. **Model Lifecycle Management**: Automated retraining and A/B testing
4. **Monitoring and Alerting**: Performance tracking and notifications

## Workflow Files

### 1. Main CI/CD Pipeline (`ci-cd-pipeline.yml`)

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main`
- Daily scheduled runs for retraining checks

**Jobs:**
1. **test-and-quality**: Code formatting, linting, type checking, unit tests
2. **data-and-model-validation**: Data schema validation, model training tests
3. **security-scan**: Dependency vulnerability scanning, security linting
4. **build-images**: Docker image building and testing
5. **deploy-staging**: Deployment to staging environment (develop branch)
6. **model-validation**: ML model validation and registration (main branch)
7. **deploy-production**: Production deployment with smoke tests
8. **setup-monitoring**: Application Insights and alerting setup
9. **cleanup-and-notify**: Resource cleanup and team notifications

### 2. Model Retraining Pipeline (`model-retraining.yml`)

**Triggers:**
- Weekly schedule (Sunday 3 AM UTC)
- Manual workflow dispatch
- Performance degradation alerts

**Jobs:**
1. **check-retraining-triggers**: Evaluate if retraining is needed
2. **retrain-model**: Collect data, train models, select best performer
3. **deploy-challenger**: Deploy challenger model for A/B testing
4. **notify-retraining**: Send notifications about retraining status

### 3. A/B Test Evaluation (`ab-test-evaluation.yml`)

**Triggers:**
- Daily schedule (6 AM UTC)
- Manual evaluation requests
- Automatic evaluation after retraining

**Jobs:**
1. **evaluate-ab-test**: Collect metrics and evaluate challenger performance
2. **promote-challenger**: Promote challenger to champion if criteria met
3. **notify-results**: Send notifications about promotion decisions

## Environment Setup

### Required Secrets

Add these secrets to your GitHub repository:

```bash
# Azure Credentials
AZURE_SUBSCRIPTION_ID=<your-subscription-id>
AZURE_TENANT_ID=<your-tenant-id>
AZURE_CLIENT_ID=<your-client-id>
AZURE_CLIENT_SECRET=<your-client-secret>
AZURE_RESOURCE_GROUP=<your-resource-group>
AZURE_ML_WORKSPACE=<your-ml-workspace>

# Azure Credentials JSON (for Azure Login action)
AZURE_CREDENTIALS='{
  "clientId": "<client-id>",
  "clientSecret": "<client-secret>",
  "subscriptionId": "<subscription-id>",
  "tenantId": "<tenant-id>"
}'

# MLflow Configuration
MLFLOW_TRACKING_URI=<your-mlflow-tracking-uri>
```

### Branch Protection Rules

Set up branch protection for `main`:

1. Require pull request reviews
2. Require status checks to pass:
   - `test-and-quality`
   - `data-and-model-validation`
   - `security-scan`
3. Require branches to be up to date
4. Restrict pushes to matching branches

## Deployment Environments

### Staging Environment
- **Trigger**: Push to `develop` branch
- **Purpose**: Integration testing and validation
- **Resources**: Azure Container Instances
- **URL Pattern**: `dynamic-pricing-*-staging-{run-number}.eastus.azurecontainer.io`

### Production Environment
- **Trigger**: Push to `main` branch (after all checks pass)
- **Purpose**: Live production serving
- **Resources**: Azure App Service + Container Instances
- **URL Pattern**: `dynamic-pricing-*-prod-{run-number}.azurewebsites.net`

## Model Lifecycle

### Champion-Challenger Strategy

1. **Champion Model**: Current production model serving 100% traffic
2. **Challenger Model**: New model serving 10% traffic during A/B test
3. **Evaluation Period**: 7 days of A/B testing
4. **Promotion Criteria**:
   - MAE improvement ≥ 3%
   - Error rate not increased by >10%
   - Response time not increased by >10%
   - Minimum 500 requests for statistical power

### Automated Retraining Triggers

1. **Scheduled**: Weekly on Sundays
2. **Performance-based**: When model performance degrades
3. **Data drift**: When significant drift detected
4. **Manual**: Via workflow dispatch

## Monitoring and Alerting

### Application Insights Integration
- API response times and error rates
- Custom metrics for ML model performance
- Automated alerts for performance degradation

### GitHub Notifications
- Automated issues created for:
  - Successful deployments
  - Failed deployments
  - Retraining completions
  - A/B test results
  - Performance alerts

### Alert Thresholds
- API response time > 5 seconds
- Error rate > 5 errors/minute
- Model MAE degradation > 10%
- Data drift > 20% of features

## Usage Guide

### Manual Deployment

```bash
# Deploy to staging
git checkout develop
git push origin develop

# Deploy to production
git checkout main
git merge develop
git push origin main
```

### Manual Retraining

```bash
# Trigger retraining workflow
gh workflow run model-retraining.yml \
  -f force_retrain=true \
  -f retrain_reason="Performance degradation detected"
```

### Manual A/B Test Evaluation

```bash
# Evaluate current A/B test
gh workflow run ab-test-evaluation.yml

# Force promote challenger
gh workflow run ab-test-evaluation.yml \
  -f force_promotion=true
```

### Checking Workflow Status

```bash
# List recent workflow runs
gh run list

# View specific workflow
gh run view <run-id>

# View logs
gh run view <run-id> --log
```

## Rollback Procedures

### Application Rollback
1. Identify last known good deployment
2. Redeploy using previous Docker image tag
3. Update MLflow model alias to previous version

### Model Rollback
```python
import mlflow

# Rollback to previous champion
client = mlflow.MlflowClient()
previous_version = client.get_model_version("dynamic_pricing_model_prod", "N-1")
client.transition_model_version_stage(
    "dynamic_pricing_model_prod",
    previous_version.version,
    "Production"
)
```

## Performance Optimization

### Build Optimization
- Docker layer caching
- Dependency caching with pip cache
- Parallel job execution where possible
- Artifact reuse between jobs

### Cost Optimization
- Automatic cleanup of staging resources
- Scheduled resource scaling
- Spot instances for training workloads

## Security Best Practices

### Code Security
- Dependency vulnerability scanning with Safety
- Static code analysis with Bandit
- Secret scanning with TruffleHog
- Container image security scanning

### Infrastructure Security
- Azure Key Vault for secrets management
- Managed identity for Azure resources
- Network security groups for container isolation
- HTTPS/TLS for all endpoints

## Troubleshooting

### Common Issues

1. **Azure Authentication Failures**
   - Verify AZURE_CREDENTIALS secret format
   - Check service principal permissions
   - Ensure subscription access

2. **MLflow Connection Issues**
   - Verify MLFLOW_TRACKING_URI
   - Check Azure ML workspace connectivity
   - Validate authentication tokens

3. **Docker Build Failures**
   - Check Dockerfile syntax
   - Verify base image availability
   - Review dependency conflicts

4. **Model Registration Failures**
   - Ensure MLflow experiment exists
   - Check model artifacts are created
   - Verify model registry permissions

### Debug Commands

```bash
# Check Azure login
az account show

# Test MLflow connection
python -c "import mlflow; print(mlflow.get_tracking_uri())"

# Validate model file
python -c "import joblib; model = joblib.load('model.pkl'); print(type(model))"

# Test API endpoint
curl -f https://your-api-endpoint/health
```

## Monitoring Dashboard

Access the CI/CD dashboard at:
- **GitHub Actions**: Repository → Actions tab
- **Azure Portal**: Resource groups → Monitoring
- **MLflow UI**: Your MLflow tracking server
- **Application Insights**: Azure portal → Application Insights

## Next Steps

1. **Enhanced Testing**: Add integration tests with test data
2. **Performance Testing**: Add load testing for production readiness
3. **Multi-region Deployment**: Extend to multiple Azure regions
4. **Advanced A/B Testing**: Implement statistical significance testing
5. **Model Explainability**: Add model interpretation in pipeline
6. **Data Quality Monitoring**: Implement automated data validation
7. **Cost Monitoring**: Add Azure cost tracking and optimization

## Support

For issues with the CI/CD pipeline:
1. Check workflow logs in GitHub Actions
2. Review Azure resource status in portal
3. Check MLflow experiment tracking
4. Contact the ML Engineering team

This completes the Phase 6 CI/CD implementation with comprehensive automation for the entire ML lifecycle.
