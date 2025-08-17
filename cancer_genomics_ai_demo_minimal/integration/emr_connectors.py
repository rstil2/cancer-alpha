#!/usr/bin/env python3
"""
EMR Connector Framework for Oncura
=======================================

Provides seamless integration with major Electronic Medical Record (EMR) systems
including Epic and Cerner through their respective APIs and authentication systems.

Author: Oncura Research Team
Date: August 2025
Version: 1.0.0
"""

import requests
import json
import base64
import hashlib
import hmac
import time
import uuid
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod
import jwt
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding

logger = logging.getLogger(__name__)

@dataclass
class EMRCredentials:
    """EMR system credentials and configuration"""
    client_id: str
    client_secret: Optional[str] = None
    base_url: str = ""
    auth_url: str = ""
    scope: str = ""
    private_key: Optional[str] = None
    certificate: Optional[str] = None

@dataclass
class PatientRecord:
    """Standardized patient record structure"""
    patient_id: str
    mrn: str
    first_name: str
    last_name: str
    date_of_birth: str
    gender: str
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[Dict[str, str]] = None

@dataclass
class OrderRecord:
    """Genomic test order record"""
    order_id: str
    patient_id: str
    ordered_by: str
    order_date: str
    test_type: str
    priority: str = "routine"
    status: str = "pending"

class BaseEMRConnector(ABC):
    """Base class for EMR connectors"""
    
    def __init__(self, credentials: EMRCredentials):
        self.credentials = credentials
        self.access_token = None
        self.token_expires_at = None
        self.session = requests.Session()
        
    @abstractmethod
    async def authenticate(self) -> bool:
        """Authenticate with the EMR system"""
        pass
        
    @abstractmethod
    async def get_patient_data(self, patient_id: str) -> Optional[PatientRecord]:
        """Retrieve patient data from EMR"""
        pass
        
    @abstractmethod
    async def post_results(self, patient_id: str, results: Dict[str, Any]) -> bool:
        """Post genomic results back to EMR"""
        pass
        
    @abstractmethod
    async def get_pending_orders(self) -> List[OrderRecord]:
        """Get pending genomic test orders"""
        pass
    
    def is_token_valid(self) -> bool:
        """Check if current access token is valid"""
        if not self.access_token or not self.token_expires_at:
            return False
        return datetime.now() < self.token_expires_at
    
    async def ensure_authenticated(self) -> bool:
        """Ensure we have a valid authentication token"""
        if not self.is_token_valid():
            return await self.authenticate()
        return True

class EpicConnector(BaseEMRConnector):
    """
    Epic EMR connector using SMART on FHIR and Epic's proprietary APIs
    
    Supports:
    - OAuth 2.0 authentication
    - FHIR R4 patient data retrieval
    - Results posting via MyChart
    - Provider notification system
    """
    
    def __init__(self, credentials: EMRCredentials):
        super().__init__(credentials)
        self.fhir_base = f"{credentials.base_url}/api/FHIR/R4"
        
    async def authenticate(self) -> bool:
        """
        Authenticate with Epic using OAuth 2.0 client credentials flow
        """
        try:
            auth_data = {
                'grant_type': 'client_credentials',
                'client_id': self.credentials.client_id,
                'client_secret': self.credentials.client_secret,
                'scope': self.credentials.scope or 'system/Patient.read system/Observation.write'
            }
            
            response = requests.post(
                f"{self.credentials.auth_url}/oauth2/token",
                data=auth_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get('access_token')
                expires_in = token_data.get('expires_in', 3600)
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)
                
                # Set authorization header for all future requests
                self.session.headers.update({
                    'Authorization': f'Bearer {self.access_token}',
                    'Content-Type': 'application/fhir+json',
                    'Accept': 'application/fhir+json'
                })
                
                logger.info("✅ Epic authentication successful")
                return True
            else:
                logger.error(f"❌ Epic authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Epic authentication error: {str(e)}")
            return False
    
    async def get_patient_data(self, patient_id: str) -> Optional[PatientRecord]:
        """
        Retrieve patient data from Epic using FHIR
        
        Args:
            patient_id: Epic patient identifier
            
        Returns:
            PatientRecord object or None if not found
        """
        try:
            if not await self.ensure_authenticated():
                return None
            
            # Get patient resource from Epic FHIR API
            response = self.session.get(f"{self.fhir_base}/Patient/{patient_id}")
            
            if response.status_code == 200:
                patient_fhir = response.json()
                
                # Extract patient data from FHIR resource
                patient_record = PatientRecord(
                    patient_id=patient_id,
                    mrn=self._extract_mrn(patient_fhir),
                    first_name=self._extract_first_name(patient_fhir),
                    last_name=self._extract_last_name(patient_fhir),
                    date_of_birth=patient_fhir.get('birthDate', ''),
                    gender=patient_fhir.get('gender', 'unknown'),
                    phone=self._extract_phone(patient_fhir),
                    email=self._extract_email(patient_fhir),
                    address=self._extract_address(patient_fhir)
                )
                
                logger.info(f"✅ Retrieved patient data for {patient_record.mrn}")
                return patient_record
            else:
                logger.error(f"❌ Failed to retrieve patient {patient_id}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error retrieving patient data: {str(e)}")
            return None
    
    async def post_results(self, patient_id: str, results: Dict[str, Any]) -> bool:
        """
        Post genomic results to Epic via FHIR Observation and DiagnosticReport
        
        Args:
            patient_id: Epic patient identifier
            results: Oncura prediction results
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not await self.ensure_authenticated():
                return False
            
            # Create FHIR Observation for the genomic result
            observation = self._create_epic_observation(patient_id, results)
            
            # Post observation to Epic
            obs_response = self.session.post(
                f"{self.fhir_base}/Observation",
                json=observation
            )
            
            if obs_response.status_code in [200, 201]:
                observation_id = obs_response.json().get('id')
                
                # Create diagnostic report
                diagnostic_report = self._create_epic_diagnostic_report(
                    patient_id, results, observation_id
                )
                
                # Post diagnostic report
                dr_response = self.session.post(
                    f"{self.fhir_base}/DiagnosticReport",
                    json=diagnostic_report
                )
                
                if dr_response.status_code in [200, 201]:
                    logger.info(f"✅ Posted results to Epic for patient {patient_id}")
                    
                    # Notify provider through Epic's messaging system
                    await self._notify_epic_provider(patient_id, results)
                    
                    return True
                else:
                    logger.error(f"❌ Failed to post diagnostic report: {dr_response.status_code}")
                    return False
            else:
                logger.error(f"❌ Failed to post observation: {obs_response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error posting results to Epic: {str(e)}")
            return False
    
    async def get_pending_orders(self) -> List[OrderRecord]:
        """
        Get pending genomic test orders from Epic
        
        Returns:
            List of pending OrderRecord objects
        """
        try:
            if not await self.ensure_authenticated():
                return []
            
            # Search for orders with specific CPT codes for genomic testing
            genomic_cpt_codes = ['81445', '81450', '81455']  # Common genomic profiling codes
            orders = []
            
            for cpt_code in genomic_cpt_codes:
                response = self.session.get(
                    f"{self.fhir_base}/ServiceRequest",
                    params={
                        'code': f'http://www.ama-assn.org/go/cpt|{cpt_code}',
                        'status': 'active',
                        '_count': 50
                    }
                )
                
                if response.status_code == 200:
                    bundle = response.json()
                    if bundle.get('entry'):
                        for entry in bundle['entry']:
                            service_request = entry['resource']
                            order = self._parse_epic_order(service_request)
                            if order:
                                orders.append(order)
            
            logger.info(f"✅ Retrieved {len(orders)} pending orders from Epic")
            return orders
            
        except Exception as e:
            logger.error(f"❌ Error retrieving orders from Epic: {str(e)}")
            return []
    
    def _extract_mrn(self, patient_fhir: Dict) -> str:
        """Extract MRN from Epic patient FHIR resource"""
        for identifier in patient_fhir.get('identifier', []):
            if identifier.get('type', {}).get('coding', [{}])[0].get('code') == 'MR':
                return identifier.get('value', '')
        return ''
    
    def _extract_first_name(self, patient_fhir: Dict) -> str:
        """Extract first name from Epic patient FHIR resource"""
        names = patient_fhir.get('name', [])
        if names:
            given = names[0].get('given', [])
            return given[0] if given else ''
        return ''
    
    def _extract_last_name(self, patient_fhir: Dict) -> str:
        """Extract last name from Epic patient FHIR resource"""
        names = patient_fhir.get('name', [])
        return names[0].get('family', '') if names else ''
    
    def _extract_phone(self, patient_fhir: Dict) -> Optional[str]:
        """Extract phone from Epic patient FHIR resource"""
        telecoms = patient_fhir.get('telecom', [])
        for telecom in telecoms:
            if telecom.get('system') == 'phone':
                return telecom.get('value')
        return None
    
    def _extract_email(self, patient_fhir: Dict) -> Optional[str]:
        """Extract email from Epic patient FHIR resource"""
        telecoms = patient_fhir.get('telecom', [])
        for telecom in telecoms:
            if telecom.get('system') == 'email':
                return telecom.get('value')
        return None
    
    def _extract_address(self, patient_fhir: Dict) -> Optional[Dict[str, str]]:
        """Extract address from Epic patient FHIR resource"""
        addresses = patient_fhir.get('address', [])
        if addresses:
            addr = addresses[0]
            return {
                'street': addr.get('line', [''])[0],
                'city': addr.get('city', ''),
                'state': addr.get('state', ''),
                'zip': addr.get('postalCode', '')
            }
        return None
    
    def _create_epic_observation(self, patient_id: str, results: Dict[str, Any]) -> Dict:
        """Create Epic-compatible FHIR Observation"""
        return {
            'resourceType': 'Observation',
            'status': 'final',
            'category': [{
                'coding': [{
                    'system': 'http://terminology.hl7.org/CodeSystem/observation-category',
                    'code': 'laboratory'
                }]
            }],
            'code': {
                'coding': [{
                    'system': 'http://loinc.org',
                    'code': '81247-9',
                    'display': 'Genomic Cancer Classification'
                }]
            },
            'subject': {'reference': f'Patient/{patient_id}'},
            'effectiveDateTime': results.get('timestamp'),
            'valueCodeableConcept': {
                'coding': [{
                    'system': 'http://canceralphasolutions.com/cancer-types',
                    'code': results.get('predicted_cancer_type'),
                    'display': f"Oncura Prediction: {results.get('predicted_cancer_type')}"
                }]
            },
            'component': [
                {
                    'code': {
                        'coding': [{
                            'system': 'http://loinc.org',
                            'code': 'LA6118-8',
                            'display': 'Confidence Score'
                        }]
                    },
                    'valueQuantity': {
                        'value': results.get('confidence_score', 0) * 100,
                        'unit': 'percent'
                    }
                }
            ]
        }
    
    def _create_epic_diagnostic_report(self, patient_id: str, results: Dict[str, Any], 
                                     observation_id: str) -> Dict:
        """Create Epic-compatible DiagnosticReport"""
        return {
            'resourceType': 'DiagnosticReport',
            'status': 'final',
            'category': [{
                'coding': [{
                    'system': 'http://terminology.hl7.org/CodeSystem/v2-0074',
                    'code': 'LAB'
                }]
            }],
            'code': {
                'coding': [{
                    'system': 'http://loinc.org',
                    'code': '81247-9',
                    'display': 'Oncura Genomic Analysis'
                }]
            },
            'subject': {'reference': f'Patient/{patient_id}'},
            'effectiveDateTime': results.get('timestamp'),
            'result': [{'reference': f'Observation/{observation_id}'}],
            'conclusion': f"Oncura AI Analysis: {results.get('predicted_cancer_type')} "
                         f"(Confidence: {results.get('confidence_score', 0):.1%})"
        }
    
    def _parse_epic_order(self, service_request: Dict) -> Optional[OrderRecord]:
        """Parse Epic ServiceRequest into OrderRecord"""
        try:
            return OrderRecord(
                order_id=service_request.get('id', ''),
                patient_id=service_request.get('subject', {}).get('reference', '').split('/')[-1],
                ordered_by=service_request.get('requester', {}).get('reference', ''),
                order_date=service_request.get('authoredOn', ''),
                test_type='genomic_cancer_classification',
                status=service_request.get('status', 'active')
            )
        except Exception:
            return None
    
    async def _notify_epic_provider(self, patient_id: str, results: Dict[str, Any]) -> bool:
        """Send notification to provider through Epic messaging"""
        try:
            # This would integrate with Epic's messaging API
            # For now, we'll create a Communication resource
            message = {
                'resourceType': 'Communication',
                'status': 'completed',
                'subject': {'reference': f'Patient/{patient_id}'},
                'sent': datetime.now().isoformat(),
                'payload': [{
                    'contentString': f"Oncura genomic results available for patient. "
                                   f"Predicted type: {results.get('predicted_cancer_type')} "
                                   f"(Confidence: {results.get('confidence_score', 0):.1%})"
                }]
            }
            
            response = self.session.post(f"{self.fhir_base}/Communication", json=message)
            return response.status_code in [200, 201]
            
        except Exception as e:
            logger.error(f"❌ Error notifying Epic provider: {str(e)}")
            return False

class CernerConnector(BaseEMRConnector):
    """
    Cerner EMR connector using SMART on FHIR and Cerner's proprietary APIs
    
    Supports:
    - OAuth 2.0 authentication  
    - FHIR R4 patient data retrieval
    - Results posting via PowerChart
    - Clinical decision support alerts
    """
    
    def __init__(self, credentials: EMRCredentials):
        super().__init__(credentials)
        self.fhir_base = f"{credentials.base_url}/fhir/r4"
        
    async def authenticate(self) -> bool:
        """
        Authenticate with Cerner using OAuth 2.0
        """
        try:
            # Cerner often uses JWT bearer token authentication
            if self.credentials.private_key:
                jwt_token = self._create_jwt_assertion()
                
                auth_data = {
                    'grant_type': 'client_credentials',
                    'client_assertion_type': 'urn:ietf:params:oauth:client-assertion-type:jwt-bearer',
                    'client_assertion': jwt_token,
                    'scope': self.credentials.scope or 'system/Patient.read system/Observation.write'
                }
            else:
                auth_data = {
                    'grant_type': 'client_credentials',
                    'client_id': self.credentials.client_id,
                    'client_secret': self.credentials.client_secret,
                    'scope': self.credentials.scope or 'system/Patient.read system/Observation.write'
                }
            
            response = requests.post(
                f"{self.credentials.auth_url}/token",
                data=auth_data,
                headers={'Content-Type': 'application/x-www-form-urlencoded'}
            )
            
            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data.get('access_token')
                expires_in = token_data.get('expires_in', 3600)
                self.token_expires_at = datetime.now() + timedelta(seconds=expires_in - 60)
                
                self.session.headers.update({
                    'Authorization': f'Bearer {self.access_token}',
                    'Content-Type': 'application/fhir+json',
                    'Accept': 'application/fhir+json'
                })
                
                logger.info("✅ Cerner authentication successful")
                return True
            else:
                logger.error(f"❌ Cerner authentication failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Cerner authentication error: {str(e)}")
            return False
    
    def _create_jwt_assertion(self) -> str:
        """Create JWT assertion for Cerner authentication"""
        now = int(time.time())
        payload = {
            'iss': self.credentials.client_id,
            'sub': self.credentials.client_id,
            'aud': self.credentials.auth_url,
            'jti': str(uuid.uuid4()),
            'exp': now + 300,  # 5 minutes
            'iat': now
        }
        
        # Sign with private key
        return jwt.encode(payload, self.credentials.private_key, algorithm='RS256')
    
    async def get_patient_data(self, patient_id: str) -> Optional[PatientRecord]:
        """
        Retrieve patient data from Cerner using FHIR
        """
        try:
            if not await self.ensure_authenticated():
                return None
            
            response = self.session.get(f"{self.fhir_base}/Patient/{patient_id}")
            
            if response.status_code == 200:
                patient_fhir = response.json()
                
                patient_record = PatientRecord(
                    patient_id=patient_id,
                    mrn=self._extract_cerner_mrn(patient_fhir),
                    first_name=self._extract_first_name(patient_fhir),
                    last_name=self._extract_last_name(patient_fhir),
                    date_of_birth=patient_fhir.get('birthDate', ''),
                    gender=patient_fhir.get('gender', 'unknown'),
                    phone=self._extract_phone(patient_fhir),
                    email=self._extract_email(patient_fhir),
                    address=self._extract_address(patient_fhir)
                )
                
                logger.info(f"✅ Retrieved Cerner patient data for {patient_record.mrn}")
                return patient_record
            else:
                logger.error(f"❌ Failed to retrieve Cerner patient {patient_id}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"❌ Error retrieving Cerner patient data: {str(e)}")
            return None
    
    async def post_results(self, patient_id: str, results: Dict[str, Any]) -> bool:
        """
        Post genomic results to Cerner PowerChart
        """
        try:
            if not await self.ensure_authenticated():
                return False
            
            # Create Cerner-compatible observation
            observation = self._create_cerner_observation(patient_id, results)
            
            response = self.session.post(f"{self.fhir_base}/Observation", json=observation)
            
            if response.status_code in [200, 201]:
                logger.info(f"✅ Posted results to Cerner for patient {patient_id}")
                
                # Create clinical decision support alert
                await self._create_cerner_cds_alert(patient_id, results)
                
                return True
            else:
                logger.error(f"❌ Failed to post Cerner observation: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error posting results to Cerner: {str(e)}")
            return False
    
    async def get_pending_orders(self) -> List[OrderRecord]:
        """
        Get pending genomic test orders from Cerner
        """
        try:
            if not await self.ensure_authenticated():
                return []
            
            response = self.session.get(
                f"{self.fhir_base}/ServiceRequest",
                params={
                    'status': 'active',
                    'category': 'http://terminology.hl7.org/CodeSystem/medicationrequest-category|community',
                    '_count': 50
                }
            )
            
            orders = []
            if response.status_code == 200:
                bundle = response.json()
                if bundle.get('entry'):
                    for entry in bundle['entry']:
                        order = self._parse_cerner_order(entry['resource'])
                        if order:
                            orders.append(order)
            
            logger.info(f"✅ Retrieved {len(orders)} pending orders from Cerner")
            return orders
            
        except Exception as e:
            logger.error(f"❌ Error retrieving Cerner orders: {str(e)}")
            return []
    
    def _extract_cerner_mrn(self, patient_fhir: Dict) -> str:
        """Extract MRN from Cerner patient FHIR resource"""
        # Cerner may use different identifier systems
        for identifier in patient_fhir.get('identifier', []):
            system = identifier.get('system', '')
            if 'MRN' in system or 'mrn' in system.lower():
                return identifier.get('value', '')
        return ''
    
    def _create_cerner_observation(self, patient_id: str, results: Dict[str, Any]) -> Dict:
        """Create Cerner-compatible FHIR Observation"""
        return {
            'resourceType': 'Observation',
            'status': 'final',
            'category': [{
                'coding': [{
                    'system': 'http://terminology.hl7.org/CodeSystem/observation-category',
                    'code': 'laboratory',
                    'display': 'Laboratory'
                }]
            }],
            'code': {
                'coding': [{
                    'system': 'http://loinc.org',
                    'code': '81247-9',
                    'display': 'Oncura Genomic Classification'
                }]
            },
            'subject': {'reference': f'Patient/{patient_id}'},
            'effectiveDateTime': results.get('timestamp'),
            'valueCodeableConcept': {
                'coding': [{
                    'system': 'http://canceralphasolutions.com/cancer-types',
                    'code': results.get('predicted_cancer_type'),
                    'display': f"Predicted: {results.get('predicted_cancer_type')}"
                }]
            },
            'interpretation': [{
                'coding': [{
                    'system': 'http://terminology.hl7.org/CodeSystem/v3-ObservationInterpretation',
                    'code': 'H' if results.get('confidence_score', 0) > 0.8 else 'N',
                    'display': 'High confidence' if results.get('confidence_score', 0) > 0.8 else 'Normal confidence'
                }]
            }]
        }
    
    def _parse_cerner_order(self, service_request: Dict) -> Optional[OrderRecord]:
        """Parse Cerner ServiceRequest into OrderRecord"""
        try:
            return OrderRecord(
                order_id=service_request.get('id', ''),
                patient_id=service_request.get('subject', {}).get('reference', '').split('/')[-1],
                ordered_by=service_request.get('requester', {}).get('reference', ''),
                order_date=service_request.get('authoredOn', ''),
                test_type='genomic_cancer_classification',
                status=service_request.get('status', 'active')
            )
        except Exception:
            return None
    
    async def _create_cerner_cds_alert(self, patient_id: str, results: Dict[str, Any]) -> bool:
        """Create clinical decision support alert in Cerner"""
        try:
            # Create CDS Hooks card for clinical decision support
            alert = {
                'resourceType': 'Flag',
                'status': 'active',
                'category': [{
                    'coding': [{
                        'system': 'http://terminology.hl7.org/CodeSystem/flag-category',
                        'code': 'clinical',
                        'display': 'Clinical'
                    }]
                }],
                'code': {
                    'coding': [{
                        'system': 'http://canceralphasolutions.com/alerts',
                        'code': 'genomic-result-available',
                        'display': 'Genomic Cancer Classification Available'
                    }]
                },
                'subject': {'reference': f'Patient/{patient_id}'},
                'period': {
                    'start': datetime.now().isoformat()
                }
            }
            
            response = self.session.post(f"{self.fhir_base}/Flag", json=alert)
            return response.status_code in [200, 201]
            
        except Exception as e:
            logger.error(f"❌ Error creating Cerner CDS alert: {str(e)}")
            return False

class EMRConnectorFactory:
    """Factory for creating appropriate EMR connectors"""
    
    @staticmethod
    def create_connector(emr_type: str, credentials: EMRCredentials) -> BaseEMRConnector:
        """
        Create EMR connector based on type
        
        Args:
            emr_type: Type of EMR system ('epic', 'cerner')
            credentials: EMR credentials
            
        Returns:
            Appropriate EMR connector instance
        """
        if emr_type.lower() == 'epic':
            return EpicConnector(credentials)
        elif emr_type.lower() == 'cerner':
            return CernerConnector(credentials)
        else:
            raise ValueError(f"Unsupported EMR type: {emr_type}")

# Example usage
async def example_emr_integration():
    """Example of EMR connector usage"""
    
    # Epic configuration
    epic_credentials = EMRCredentials(
        client_id="cancer-alpha-epic-client",
        client_secret="your-epic-secret",
        base_url="https://apporchard.epic.com/interconnect-aocurprd-oauth",
        auth_url="https://apporchard.epic.com/interconnect-aocurprd-oauth",
        scope="system/Patient.read system/Observation.write"
    )
    
    # Create Epic connector
    epic = EMRConnectorFactory.create_connector('epic', epic_credentials)
    
    # Authenticate
    if await epic.authenticate():
        print("✅ Epic authentication successful")
        
        # Get patient data
        patient = await epic.get_patient_data("patient-123")
        if patient:
            print(f"✅ Retrieved patient: {patient.first_name} {patient.last_name}")
        
        # Get pending orders
        orders = await epic.get_pending_orders()
        print(f"✅ Found {len(orders)} pending orders")
        
        # Example results posting
        sample_results = {
            'prediction_id': 'pred-12345',
            'predicted_cancer_type': 'BRCA',
            'confidence_score': 0.95,
            'timestamp': datetime.now().isoformat(),
            'biological_insights': ['High confidence BRCA prediction']
        }
        
        if await epic.post_results("patient-123", sample_results):
            print("✅ Results posted to Epic successfully")

if __name__ == "__main__":
    import asyncio
    asyncio.run(example_emr_integration())
