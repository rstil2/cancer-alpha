# Cancer Alpha Hospital Deployment Guide
## Complete "Plug and Play" Integration for Healthcare Systems

**Version:** 1.0.0  
**Date:** August 2025  
**Author:** Cancer Alpha Research Team  

---

## 🏥 Overview

This guide provides step-by-step instructions for deploying Cancer Alpha as a fully integrated "plug and play" solution in hospital environments. The system provides 95.0% accuracy cancer genomics classification with seamless EHR integration, clinical workflow automation, and regulatory compliance.

### What's Included
- **FHIR R4 Integration** - Seamless EHR connectivity
- **Epic/Cerner Connectors** - Direct EMR integration
- **Clinical Workflow API** - Order management and notifications
- **Hospital Authentication** - SSO, LDAP, and role-based access
- **Compliance Framework** - HIPAA auditing and security

---

## 📋 Prerequisites

### System Requirements
- **Operating System**: Linux (Ubuntu 20.04+ recommended) or Docker environment
- **Memory**: Minimum 8GB RAM (16GB recommended for production)
- **CPU**: 4+ cores recommended
- **Storage**: 50GB+ available space
- **Network**: Outbound HTTPS access for model updates and licensing

### Hospital IT Requirements
- **Network Access**: Ability to create internal DNS entries and firewall rules
- **Database Access**: Connection to hospital patient database (optional)
- **EMR Integration**: Epic/Cerner API credentials and FHIR endpoints
- **Authentication**: LDAP/AD credentials or SSO configuration
- **Security**: SSL certificates for HTTPS endpoints

---

## 🚀 Quick Start Deployment

### Option 1: Docker Compose (Recommended)

1. **Clone the Repository**
```bash
git clone https://github.com/your-org/cancer-alpha.git
cd cancer-alpha/cancer_genomics_ai_demo_minimal
```

2. **Configure Environment**
```bash
# Copy example configuration
cp .env.example .env.hospital

# Edit configuration for your hospital
nano .env.hospital
```

3. **Deploy with Docker**
```bash
# Start all services
docker-compose --env-file .env.hospital up -d

# Check service health
docker-compose ps
curl http://localhost:8000/health
```

### Option 2: Native Installation

1. **Install Dependencies**
```bash
# Install Python dependencies
pip install -r requirements_hospital.txt

# Install system dependencies (Ubuntu)
sudo apt-get update
sudo apt-get install -y redis-server nginx postgresql
```

2. **Configure Services**
```bash
# Set up configuration
python setup_hospital.py --interactive

# Start services
./start_hospital_deployment.sh
```

---

## ⚙️ Configuration

### Environment Variables (.env.hospital)

```bash
# Hospital Identification
HOSPITAL_ID=main_hospital
HOSPITAL_NAME="Main Medical Center"
HOSPITAL_DOMAIN=mainmedical.com

# Cancer Alpha API
CANCER_ALPHA_API_KEY=your-production-api-key
CANCER_ALPHA_BASE_URL=http://localhost:8000

# Database Configuration
DATABASE_URL=postgresql://cancer_alpha:password@localhost:5432/cancer_alpha_hospital
REDIS_URL=redis://localhost:6379/0

# Authentication
JWT_SECRET=your-super-secure-jwt-secret-here
LDAP_SERVER=ldap.mainmedical.com
LDAP_BASE_DN=ou=users,dc=mainmedical,dc=com
LDAP_BIND_USER=cn=cancer-alpha,ou=service-accounts,dc=mainmedical,dc=com
LDAP_BIND_PASSWORD=ldap-service-password

# SSO Configuration (Optional)
SSO_ENABLED=true
SAML_METADATA_URL=https://sso.mainmedical.com/metadata
OIDC_DISCOVERY_URL=https://sso.mainmedical.com/.well-known/openid_configuration

# EMR Integration
EMR_TYPE=epic  # Options: epic, cerner
EPIC_CLIENT_ID=cancer-alpha-client-id
EPIC_CLIENT_SECRET=your-epic-client-secret
EPIC_BASE_URL=https://apporchard.epic.com/interconnect-aocurprd-oauth
EPIC_AUTH_URL=https://apporchard.epic.com/interconnect-aocurprd-oauth

# FHIR Configuration
FHIR_BASE_URL=https://fhir.mainmedical.com/R4
FHIR_CLIENT_ID=cancer-alpha-fhir-client
FHIR_CLIENT_SECRET=fhir-client-secret

# Clinical Workflow
WORKFLOW_API_KEY=clinical-workflow-key-2025
NOTIFICATION_EMAIL_SMTP=smtp.mainmedical.com
NOTIFICATION_EMAIL_PORT=587
NOTIFICATION_EMAIL_USER=cancer-alpha@mainmedical.com
NOTIFICATION_EMAIL_PASS=email-password

# Security & Compliance
HIPAA_AUDIT_ENABLED=true
AUDIT_LOG_PATH=/var/log/cancer-alpha/audit.log
ENCRYPTION_KEY=your-32-character-encryption-key
SSL_CERT_PATH=/etc/ssl/certs/cancer-alpha.crt
SSL_KEY_PATH=/etc/ssl/private/cancer-alpha.key

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
LOG_LEVEL=INFO
```

---

## 🔗 EMR Integration Setup

### Epic Integration

1. **Register Application with Epic**
   - Contact Epic to register Cancer Alpha as a SMART on FHIR application
   - Obtain client credentials and sandbox access
   - Configure redirect URIs and scopes

2. **Configure Epic Connector**
```python
# config/epic_config.py
EPIC_CONFIG = {
    "client_id": "cancer-alpha-epic-client",
    "client_secret": "your-epic-secret",
    "base_url": "https://apporchard.epic.com/interconnect-aocurprd-oauth",
    "auth_url": "https://apporchard.epic.com/interconnect-aocurprd-oauth",
    "scope": "system/Patient.read system/Observation.write system/DiagnosticReport.write",
    "fhir_version": "R4"
}
```

3. **Test Epic Connection**
```bash
# Verify Epic connectivity
python test_epic_connection.py --config config/epic_config.py

# Expected output:
# ✅ Epic authentication successful
# ✅ FHIR endpoint accessible
# ✅ Patient data retrieval working
# ✅ Results posting functional
```

### Cerner Integration

1. **Obtain Cerner Credentials**
   - Register with Cerner's SMART on FHIR platform
   - Generate client certificate for JWT authentication
   - Configure PowerChart integration

2. **Configure Cerner Connector**
```python
# config/cerner_config.py
CERNER_CONFIG = {
    "client_id": "cancer-alpha-cerner-client",
    "private_key_path": "/etc/ssl/private/cerner-client.key",
    "certificate_path": "/etc/ssl/certs/cerner-client.crt",
    "base_url": "https://fhir-open.sandboxcerner.com/dstu2/0b8a0111-e8e6-4c26-a91c-5069cbc6b1ca",
    "auth_url": "https://authorization.sandboxcerner.com/tenants/0b8a0111-e8e6-4c26-a91c-5069cbc6b1ca/protocols/oauth2/profiles/smart-v1/token"
}
```

3. **Test Cerner Connection**
```bash
# Verify Cerner connectivity
python test_cerner_connection.py --config config/cerner_config.py
```

---

## 🔐 Authentication Setup

### LDAP/Active Directory Integration

1. **Configure LDAP Connection**
```yaml
# config/ldap_config.yml
ldap:
  server: ldap.mainmedical.com
  port: 389
  use_ssl: false
  start_tls: true
  base_dn: "ou=users,dc=mainmedical,dc=com"
  bind_dn: "cn=cancer-alpha,ou=service-accounts,dc=mainmedical,dc=com"
  bind_password: "ldap-service-password"
  user_search:
    base: "ou=users,dc=mainmedical,dc=com"
    filter: "(uid={username})"
    attributes:
      - cn
      - mail
      - givenName
      - sn
      - employeeID
      - department
      - title
```

2. **Test LDAP Authentication**
```bash
# Test LDAP connectivity
python test_ldap_auth.py --username testuser --password testpass

# Expected output:
# ✅ LDAP connection successful
# ✅ User authentication successful
# ✅ User attributes retrieved
# User: Dr. John Doe (john.doe@mainmedical.com)
# Role: Physician, Department: Oncology
```

### SSO Configuration

#### SAML 2.0 Setup
```xml
<!-- saml_config.xml -->
<saml:EntityDescriptor xmlns:saml="urn:oasis:names:tc:SAML:2.0:metadata" 
                      entityID="cancer-alpha-mainmedical">
  <saml:SPSSODescriptor protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol">
    <saml:AssertionConsumerService 
        Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST"
        Location="https://cancer-alpha.mainmedical.com/sso/saml/acs"
        index="0" />
  </saml:SPSSODescriptor>
</saml:EntityDescriptor>
```

#### OpenID Connect Setup
```json
{
  "client_id": "cancer-alpha-oidc-client",
  "client_secret": "oidc-client-secret",
  "discovery_url": "https://sso.mainmedical.com/.well-known/openid_configuration",
  "scopes": ["openid", "profile", "email", "groups"],
  "redirect_uri": "https://cancer-alpha.mainmedical.com/sso/oidc/callback"
}
```

---

## 📊 Clinical Workflow Configuration

### Order Management Setup

1. **Configure Test Orders**
```python
# config/clinical_config.py
GENOMIC_TEST_CODES = {
    "cancer_genomics_classification": {
        "cpt_code": "81445",
        "description": "Cancer Alpha Genomic Classification",
        "turnaround_time_hours": 2,
        "priority_levels": ["routine", "urgent", "stat"]
    }
}

NOTIFICATION_TEMPLATES = {
    "order_received": {
        "subject": "Genomic Test Order Received - {patient_mrn}",
        "template": "order_received_template.html"
    },
    "results_available": {
        "subject": "URGENT: Cancer Alpha Results Available - {patient_mrn}",
        "template": "results_available_template.html"
    }
}
```

2. **Set Up Clinical Decision Support Rules**
```python
# config/cds_rules.py
CDS_RULES = [
    {
        "rule_id": "high_confidence_brca",
        "conditions": {
            "predicted_cancer_type": "BRCA",
            "confidence_score": {">=": 0.95}
        },
        "actions": [
            {
                "type": "create_alert",
                "severity": "critical",
                "title": "High Confidence BRCA Prediction",
                "recommendations": [
                    "Consider immediate oncology consultation",
                    "Evaluate for BRCA1/BRCA2 genetic testing",
                    "Review family history of breast/ovarian cancer"
                ]
            }
        ]
    }
]
```

### Provider Notification Setup

1. **Configure SMTP Settings**
```bash
# Email notification configuration
SMTP_SERVER=smtp.mainmedical.com
SMTP_PORT=587
SMTP_USE_TLS=true
SMTP_USERNAME=cancer-alpha@mainmedical.com
SMTP_PASSWORD=secure-email-password
```

2. **Set Up Provider Directory**
```json
{
  "providers": [
    {
      "provider_id": "DOC001",
      "npi": "1234567890",
      "name": "Dr. John Smith",
      "email": "john.smith@mainmedical.com",
      "department": "Oncology",
      "specialties": ["Medical Oncology"],
      "notification_preferences": {
        "email": true,
        "emr_message": true,
        "sms": false
      }
    }
  ]
}
```

---

## 🛡️ Security & Compliance

### HIPAA Compliance Configuration

1. **Enable Audit Logging**
```yaml
# config/audit_config.yml
audit:
  enabled: true
  log_file: /var/log/cancer-alpha/audit.log
  log_format: json
  log_level: INFO
  events:
    - user_authentication
    - patient_data_access
    - result_viewing
    - data_export
    - configuration_changes
  retention_days: 2555  # 7 years as required by HIPAA
```

2. **Configure Data Encryption**
```python
# config/encryption_config.py
ENCRYPTION_CONFIG = {
    "algorithm": "AES-256-GCM",
    "key_derivation": "PBKDF2",
    "key_iterations": 100000,
    "encrypt_patient_data": True,
    "encrypt_results": True,
    "encrypt_logs": True
}
```

3. **Set Up SSL/TLS**
```bash
# Generate SSL certificates (production should use CA-signed certificates)
openssl genrsa -out cancer-alpha.key 2048
openssl req -new -key cancer-alpha.key -out cancer-alpha.csr
openssl x509 -req -days 365 -in cancer-alpha.csr -signkey cancer-alpha.key -out cancer-alpha.crt

# Configure nginx SSL
sudo cp cancer-alpha.crt /etc/ssl/certs/
sudo cp cancer-alpha.key /etc/ssl/private/
```

### Access Control Configuration

```yaml
# config/rbac_config.yml
roles:
  admin:
    permissions:
      - read_patient_data
      - write_patient_data
      - order_tests
      - view_results
      - approve_results
      - manage_users
      - view_reports
      - export_data
  
  physician:
    permissions:
      - read_patient_data
      - write_patient_data
      - order_tests
      - view_results
      - approve_results
      - view_reports
  
  nurse:
    permissions:
      - read_patient_data
      - view_results
      - view_reports

  lab_tech:
    permissions:
      - read_patient_data
      - view_results
      - order_tests
```

---

## 🚀 Deployment Steps

### Step 1: Pre-Deployment Preparation

1. **System Requirements Check**
```bash
# Run system compatibility check
python check_system_requirements.py

# Expected output:
# ✅ Operating system compatible
# ✅ Memory requirements met (16GB available)
# ✅ CPU requirements met (8 cores available)
# ✅ Storage requirements met (100GB available)
# ✅ Network connectivity verified
```

2. **Security Hardening**
```bash
# Apply security hardening
sudo ./scripts/security_hardening.sh

# This script:
# - Configures firewall rules
# - Sets up SSL certificates
# - Configures secure file permissions
# - Enables audit logging
# - Sets up intrusion detection
```

### Step 2: Core Deployment

1. **Deploy Core Services**
```bash
# Deploy using Docker Compose
docker-compose --env-file .env.hospital up -d

# Services started:
# ✅ cancer-alpha-api (port 8000)
# ✅ clinical-workflow-api (port 8001)
# ✅ hospital-auth (port 8002)
# ✅ redis (port 6379)
# ✅ postgresql (port 5432)
# ✅ nginx (ports 80, 443)
```

2. **Initialize Database**
```bash
# Run database migrations
python manage.py migrate

# Create initial admin user
python manage.py create_admin_user --username admin --email admin@mainmedical.com
```

3. **Load Configuration**
```bash
# Load hospital-specific configuration
python manage.py load_hospital_config --file config/hospital_config.json

# Import provider directory
python manage.py import_providers --file config/provider_directory.json
```

### Step 3: Integration Setup

1. **Configure EMR Integration**
```bash
# Test Epic connection
python test_integrations.py --emr epic

# Test FHIR endpoints
python test_integrations.py --fhir

# Expected output:
# ✅ Epic authentication successful
# ✅ Patient data retrieval working
# ✅ Results posting functional
# ✅ FHIR R4 compliance verified
```

2. **Test Authentication**
```bash
# Test LDAP authentication
python test_integrations.py --auth ldap --username testuser

# Test SSO authentication
python test_integrations.py --auth sso --provider saml
```

3. **Verify Clinical Workflow**
```bash
# Create test order
python test_workflow.py --create-order --patient-mrn TEST001

# Process test order
python test_workflow.py --process-order --order-id CA-20250817-12345678

# Verify notifications
python test_workflow.py --verify-notifications
```

### Step 4: Go-Live Checklist

- [ ] Core services running and healthy
- [ ] EMR integration tested with real patient data
- [ ] Authentication working with hospital directory
- [ ] Clinical workflow tested end-to-end
- [ ] Audit logging enabled and functional
- [ ] SSL certificates configured and valid
- [ ] Firewall rules configured
- [ ] Backup procedures tested
- [ ] Disaster recovery plan verified
- [ ] Staff training completed
- [ ] Go-live approval from hospital IT security

---

## 📈 Monitoring & Maintenance

### Health Monitoring

1. **Service Health Checks**
```bash
# Check all services
./scripts/health_check.sh

# Individual service checks
curl https://cancer-alpha.mainmedical.com/health
curl https://cancer-alpha.mainmedical.com/workflow/status
curl https://cancer-alpha.mainmedical.com/auth/health
```

2. **Performance Monitoring**
```bash
# View Prometheus metrics
curl https://cancer-alpha.mainmedical.com/metrics

# Access Grafana dashboard
open https://cancer-alpha.mainmedical.com/grafana
```

### Log Management

1. **Application Logs**
```bash
# View application logs
tail -f /var/log/cancer-alpha/application.log

# View audit logs
tail -f /var/log/cancer-alpha/audit.log

# View integration logs
tail -f /var/log/cancer-alpha/integration.log
```

2. **Log Rotation**
```bash
# Configure logrotate
sudo cp config/logrotate.conf /etc/logrotate.d/cancer-alpha
```

### Backup & Recovery

1. **Database Backup**
```bash
# Daily database backup
pg_dump cancer_alpha_hospital > backup_$(date +%Y%m%d).sql

# Restore from backup
psql cancer_alpha_hospital < backup_20250817.sql
```

2. **Configuration Backup**
```bash
# Backup configuration
tar -czf cancer-alpha-config-$(date +%Y%m%d).tar.gz config/ .env.hospital

# Restore configuration
tar -xzf cancer-alpha-config-20250817.tar.gz
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. EMR Connection Failed
```bash
# Symptoms: Epic/Cerner authentication failing
# Check: Credentials and network connectivity
curl -v https://apporchard.epic.com/interconnect-aocurprd-oauth/oauth2/token

# Solution: Verify client credentials and firewall rules
```

#### 2. LDAP Authentication Issues
```bash
# Symptoms: Users cannot log in via LDAP
# Check: LDAP connectivity and credentials
ldapsearch -x -H ldap://ldap.mainmedical.com -D "cn=cancer-alpha,ou=service-accounts,dc=mainmedical,dc=com" -W

# Solution: Verify LDAP bind user and search base
```

#### 3. SSL Certificate Problems
```bash
# Symptoms: HTTPS not working or certificate warnings
# Check: Certificate validity
openssl x509 -in /etc/ssl/certs/cancer-alpha.crt -text -noout

# Solution: Renew or reconfigure certificates
```

#### 4. Database Connection Issues
```bash
# Symptoms: Database connection errors
# Check: PostgreSQL status and connectivity
sudo systemctl status postgresql
psql -h localhost -U cancer_alpha -d cancer_alpha_hospital

# Solution: Restart PostgreSQL or check credentials
```

### Log Analysis

```bash
# Search for authentication errors
grep "authentication failed" /var/log/cancer-alpha/application.log

# Search for EMR integration errors
grep "EMR.*error" /var/log/cancer-alpha/integration.log

# Search for FHIR errors
grep "FHIR.*error" /var/log/cancer-alpha/application.log

# Search for audit trail issues
grep "audit.*error" /var/log/cancer-alpha/audit.log
```

---

## 📞 Support & Licensing

### Technical Support

- **Email**: support@canceralphasolutions.com
- **Phone**: 1-800-CANCER-AI
- **Portal**: https://support.canceralphasolutions.com
- **Emergency**: 24/7 emergency support available for production issues

### Licensing & Compliance

- **Patent**: Provisional Application No. 63/847,316
- **Commercial License**: Required for hospital deployment
- **Compliance**: HIPAA, SOC 2 Type II, FDA 510(k) pathway documentation available
- **Audit Support**: Assistance with regulatory audits and compliance reviews

### Training & Professional Services

- **Implementation Services**: On-site deployment assistance
- **Training Programs**: Clinical staff training and certification
- **Custom Integration**: Specialized EMR integration services
- **Ongoing Support**: 24/7 monitoring and maintenance packages

---

## 📝 Appendices

### Appendix A: Network Requirements

| Service | Port | Protocol | Description |
|---------|------|----------|-------------|
| Cancer Alpha API | 8000 | HTTPS | Main prediction API |
| Clinical Workflow | 8001 | HTTPS | Workflow management |
| Authentication | 8002 | HTTPS | User authentication |
| FHIR Endpoint | 8003 | HTTPS | FHIR R4 interface |
| Monitoring | 9090 | HTTPS | Prometheus metrics |
| Dashboard | 3000 | HTTPS | Grafana dashboard |
| Database | 5432 | TCP | PostgreSQL |
| Cache | 6379 | TCP | Redis |

### Appendix B: Required Permissions

#### Epic Permissions
- `system/Patient.read`
- `system/Observation.write`
- `system/DiagnosticReport.write`
- `system/ServiceRequest.read`
- `system/Communication.write`

#### Cerner Permissions
- `system/Patient.read`
- `system/Observation.write`
- `system/DiagnosticReport.write`
- `system/Flag.write`

### Appendix C: HIPAA Compliance Checklist

- [ ] Data encryption at rest and in transit
- [ ] User access controls and authentication
- [ ] Audit logging of all PHI access
- [ ] Data backup and recovery procedures
- [ ] Incident response plan
- [ ] Business associate agreements
- [ ] Staff training documentation
- [ ] Risk assessment completed
- [ ] Compliance monitoring procedures

---

**Document Version:** 1.0.0  
**Last Updated:** August 17, 2025  
**Next Review:** February 17, 2026  

For the most current version of this guide and additional resources, visit:  
**https://docs.canceralphasolutions.com/hospital-deployment**
