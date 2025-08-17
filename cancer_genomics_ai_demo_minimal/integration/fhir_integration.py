#!/usr/bin/env python3
"""
FHIR R4 Integration Module for Cancer Alpha
==========================================

Provides seamless integration with Electronic Health Records (EHR) systems
through FHIR R4 standard for patient data exchange, genomic observations,
and diagnostic report generation.

Author: Cancer Alpha Research Team
Date: August 2025
Version: 1.0.0
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import json
import uuid
import logging
from dataclasses import dataclass
from fhir.resources.patient import Patient
from fhir.resources.observation import Observation
from fhir.resources.diagnosticreport import DiagnosticReport
from fhir.resources.specimen import Specimen
from fhir.resources.organization import Organization
from fhir.resources.practitioner import Practitioner
from fhir.resources.coding import Coding
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.reference import Reference
from fhir.resources.identifier import Identifier
from fhir.resources.humanname import HumanName
from fhir.resources.contactpoint import ContactPoint
from fhir.resources.address import Address

logger = logging.getLogger(__name__)

@dataclass
class GenomicPrediction:
    """Data structure for Cancer Alpha predictions"""
    prediction_id: str
    predicted_cancer_type: str
    confidence_score: float
    class_probabilities: Dict[str, float]
    processing_time_ms: float
    biological_insights: List[str]
    shap_explanations: Optional[Dict]
    timestamp: str

@dataclass
class PatientGenomicData:
    """Patient genomic data structure"""
    patient_id: str
    patient_mrn: str
    genomic_features: List[float]
    specimen_id: Optional[str] = None
    collected_date: Optional[str] = None
    test_ordered_by: Optional[str] = None

class FHIRCancerAlphaIntegration:
    """
    FHIR R4 integration class for Cancer Alpha genomic predictions
    
    Provides methods to:
    - Map patient data to FHIR Patient resources
    - Create genomic observations
    - Generate diagnostic reports
    - Post results to EHR systems
    """
    
    def __init__(self, organization_id: str = "cancer-alpha-solutions"):
        self.organization_id = organization_id
        self.system_url = "http://canceralphasolutions.com/fhir"
        
    def create_patient_resource(self, patient_data: Dict[str, Any]) -> Patient:
        """
        Create FHIR Patient resource from hospital patient data
        
        Args:
            patient_data: Dictionary containing patient demographics
            
        Returns:
            FHIR Patient resource
        """
        try:
            # Create patient identifier
            identifiers = []
            if patient_data.get('mrn'):
                identifiers.append(Identifier(**{
                    "use": "usual",
                    "type": CodeableConcept(**{
                        "coding": [Coding(**{
                            "system": "http://terminology.hl7.org/CodeSystem/v2-0203",
                            "code": "MR",
                            "display": "Medical Record Number"
                        })]
                    }),
                    "system": f"{self.system_url}/patient-mrn",
                    "value": patient_data['mrn']
                }))
            
            # Create patient name
            names = []
            if patient_data.get('first_name') or patient_data.get('last_name'):
                names.append(HumanName(**{
                    "use": "official",
                    "family": patient_data.get('last_name', ''),
                    "given": [patient_data.get('first_name', '')]
                }))
            
            # Create contact info
            telecoms = []
            if patient_data.get('phone'):
                telecoms.append(ContactPoint(**{
                    "system": "phone",
                    "value": patient_data['phone'],
                    "use": "mobile"
                }))
            
            if patient_data.get('email'):
                telecoms.append(ContactPoint(**{
                    "system": "email",
                    "value": patient_data['email'],
                    "use": "home"
                }))
            
            # Create address
            addresses = []
            if any(patient_data.get(field) for field in ['street', 'city', 'state', 'zip']):
                addresses.append(Address(**{
                    "use": "home",
                    "line": [patient_data.get('street', '')],
                    "city": patient_data.get('city', ''),
                    "state": patient_data.get('state', ''),
                    "postalCode": patient_data.get('zip', '')
                }))
            
            patient = Patient(**{
                "id": patient_data.get('patient_id', str(uuid.uuid4())),
                "identifier": identifiers,
                "active": True,
                "name": names,
                "telecom": telecoms,
                "gender": patient_data.get('gender', 'unknown').lower(),
                "birthDate": patient_data.get('birth_date'),
                "address": addresses
            })
            
            logger.info(f"✅ Created FHIR Patient resource for MRN: {patient_data.get('mrn')}")
            return patient
            
        except Exception as e:
            logger.error(f"❌ Error creating Patient resource: {str(e)}")
            raise
    
    def create_genomic_observation(self, 
                                 patient_id: str,
                                 prediction: GenomicPrediction,
                                 specimen_id: Optional[str] = None) -> Observation:
        """
        Create FHIR Observation for genomic cancer prediction
        
        Args:
            patient_id: FHIR Patient ID
            prediction: Cancer Alpha prediction results
            specimen_id: Optional specimen identifier
            
        Returns:
            FHIR Observation resource
        """
        try:
            # Create observation components for each cancer type probability
            components = []
            for cancer_type, probability in prediction.class_probabilities.items():
                components.append({
                    "code": CodeableConcept(**{
                        "coding": [Coding(**{
                            "system": "http://loinc.org",
                            "code": "LA6115-4",
                            "display": f"Cancer Type Probability - {cancer_type}"
                        })]
                    }),
                    "valueQuantity": {
                        "value": round(probability * 100, 2),
                        "unit": "percent",
                        "system": "http://unitsofmeasure.org",
                        "code": "%"
                    }
                })
            
            # Add confidence score component
            components.append({
                "code": CodeableConcept(**{
                    "coding": [Coding(**{
                        "system": "http://loinc.org",
                        "code": "LA6118-8",
                        "display": "Prediction Confidence Score"
                    })]
                }),
                "valueQuantity": {
                    "value": round(prediction.confidence_score * 100, 2),
                    "unit": "percent",
                    "system": "http://unitsofmeasure.org",
                    "code": "%"
                }
            })
            
            observation = Observation(**{
                "id": prediction.prediction_id,
                "status": "final",
                "category": [CodeableConcept(**{
                    "coding": [Coding(**{
                        "system": "http://terminology.hl7.org/CodeSystem/observation-category",
                        "code": "laboratory",
                        "display": "Laboratory"
                    })]
                })],
                "code": CodeableConcept(**{
                    "coding": [Coding(**{
                        "system": "http://loinc.org",
                        "code": "81247-9",
                        "display": "Master HL7 genetic variant file"
                    })]
                }),
                "subject": Reference(**{"reference": f"Patient/{patient_id}"}),
                "effectiveDateTime": prediction.timestamp,
                "valueCodeableConcept": CodeableConcept(**{
                    "coding": [Coding(**{
                        "system": f"{self.system_url}/cancer-types",
                        "code": prediction.predicted_cancer_type,
                        "display": f"Predicted Cancer Type: {prediction.predicted_cancer_type}"
                    })]
                }),
                "component": components,
                "note": [{
                    "text": f"AI Prediction generated by Cancer Alpha system. "
                           f"Processing time: {prediction.processing_time_ms}ms. "
                           f"Biological insights: {'; '.join(prediction.biological_insights[:3])}"
                }]
            })
            
            if specimen_id:
                observation.specimen = Reference(**{"reference": f"Specimen/{specimen_id}"})
            
            logger.info(f"✅ Created genomic Observation for prediction: {prediction.prediction_id}")
            return observation
            
        except Exception as e:
            logger.error(f"❌ Error creating genomic Observation: {str(e)}")
            raise
    
    def create_diagnostic_report(self,
                               patient_id: str,
                               prediction: GenomicPrediction,
                               practitioner_id: Optional[str] = None,
                               observations: List[str] = None) -> DiagnosticReport:
        """
        Create FHIR DiagnosticReport for Cancer Alpha genomic analysis
        
        Args:
            patient_id: FHIR Patient ID
            prediction: Cancer Alpha prediction results
            practitioner_id: Ordering physician ID
            observations: List of observation IDs to include
            
        Returns:
            FHIR DiagnosticReport resource
        """
        try:
            # Create result references
            result_references = []
            if observations:
                result_references = [
                    Reference(**{"reference": f"Observation/{obs_id}"})
                    for obs_id in observations
                ]
            
            # Create conclusion text with biological insights
            conclusion_parts = [
                f"Cancer Alpha AI Analysis Results:",
                f"Predicted Cancer Type: {prediction.predicted_cancer_type}",
                f"Confidence Score: {prediction.confidence_score:.1%}",
                "",
                "Biological Insights:",
            ]
            conclusion_parts.extend([f"• {insight}" for insight in prediction.biological_insights])
            
            if prediction.shap_explanations and 'top_features' in prediction.shap_explanations:
                conclusion_parts.append("\nTop Contributing Features:")
                for feature, importance in list(prediction.shap_explanations['top_features'].items())[:5]:
                    conclusion_parts.append(f"• {feature}: {importance:.4f}")
            
            conclusion_text = "\n".join(conclusion_parts)
            
            diagnostic_report = DiagnosticReport(**{
                "id": f"dr-{prediction.prediction_id}",
                "status": "final",
                "category": [CodeableConcept(**{
                    "coding": [Coding(**{
                        "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                        "code": "LAB",
                        "display": "Laboratory"
                    })]
                })],
                "code": CodeableConcept(**{
                    "coding": [Coding(**{
                        "system": "http://loinc.org",
                        "code": "81247-9",
                        "display": "Genomic Cancer Classification Report"
                    })]
                }),
                "subject": Reference(**{"reference": f"Patient/{patient_id}"}),
                "effectiveDateTime": prediction.timestamp,
                "issued": datetime.now(timezone.utc).isoformat(),
                "performer": [Reference(**{
                    "reference": f"Organization/{self.organization_id}",
                    "display": "Cancer Alpha Solutions"
                })],
                "result": result_references,
                "conclusion": conclusion_text,
                "conclusionCode": [CodeableConcept(**{
                    "coding": [Coding(**{
                        "system": f"{self.system_url}/cancer-types",
                        "code": prediction.predicted_cancer_type,
                        "display": f"AI Predicted: {prediction.predicted_cancer_type}"
                    })]
                })]
            })
            
            if practitioner_id:
                diagnostic_report.performer.append(
                    Reference(**{"reference": f"Practitioner/{practitioner_id}"})
                )
            
            logger.info(f"✅ Created DiagnosticReport for prediction: {prediction.prediction_id}")
            return diagnostic_report
            
        except Exception as e:
            logger.error(f"❌ Error creating DiagnosticReport: {str(e)}")
            raise
    
    def create_specimen_resource(self,
                               patient_id: str,
                               specimen_data: Dict[str, Any]) -> Specimen:
        """
        Create FHIR Specimen resource for genomic sample
        
        Args:
            patient_id: FHIR Patient ID
            specimen_data: Specimen information
            
        Returns:
            FHIR Specimen resource
        """
        try:
            specimen = Specimen(**{
                "id": specimen_data.get('specimen_id', str(uuid.uuid4())),
                "identifier": [Identifier(**{
                    "system": f"{self.system_url}/specimen-id",
                    "value": specimen_data.get('specimen_id', str(uuid.uuid4()))
                })],
                "status": "available",
                "type": CodeableConcept(**{
                    "coding": [Coding(**{
                        "system": "http://snomed.info/sct",
                        "code": specimen_data.get('specimen_type_code', '119376003'),
                        "display": specimen_data.get('specimen_type', 'Tissue specimen')
                    })]
                }),
                "subject": Reference(**{"reference": f"Patient/{patient_id}"}),
                "collection": {
                    "collectedDateTime": specimen_data.get('collected_date'),
                    "bodySite": CodeableConcept(**{
                        "coding": [Coding(**{
                            "system": "http://snomed.info/sct",
                            "code": specimen_data.get('body_site_code', '123037004'),
                            "display": specimen_data.get('body_site', 'Body structure')
                        })]
                    })
                }
            })
            
            logger.info(f"✅ Created Specimen resource: {specimen_data.get('specimen_id')}")
            return specimen
            
        except Exception as e:
            logger.error(f"❌ Error creating Specimen resource: {str(e)}")
            raise
    
    def bundle_genomic_resources(self,
                               patient: Patient,
                               observation: Observation,
                               diagnostic_report: DiagnosticReport,
                               specimen: Optional[Specimen] = None) -> Dict[str, Any]:
        """
        Create FHIR Bundle containing all genomic resources
        
        Args:
            patient: Patient resource
            observation: Genomic observation
            diagnostic_report: Diagnostic report
            specimen: Optional specimen resource
            
        Returns:
            FHIR Bundle as dictionary
        """
        try:
            entries = [
                {
                    "resource": patient.dict(),
                    "request": {
                        "method": "PUT",
                        "url": f"Patient/{patient.id}"
                    }
                },
                {
                    "resource": observation.dict(),
                    "request": {
                        "method": "PUT",
                        "url": f"Observation/{observation.id}"
                    }
                },
                {
                    "resource": diagnostic_report.dict(),
                    "request": {
                        "method": "PUT",
                        "url": f"DiagnosticReport/{diagnostic_report.id}"
                    }
                }
            ]
            
            if specimen:
                entries.append({
                    "resource": specimen.dict(),
                    "request": {
                        "method": "PUT",
                        "url": f"Specimen/{specimen.id}"
                    }
                })
            
            bundle = {
                "resourceType": "Bundle",
                "id": str(uuid.uuid4()),
                "type": "transaction",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "entry": entries
            }
            
            logger.info(f"✅ Created FHIR Bundle with {len(entries)} resources")
            return bundle
            
        except Exception as e:
            logger.error(f"❌ Error creating FHIR Bundle: {str(e)}")
            raise
    
    def validate_fhir_resource(self, resource: Any) -> bool:
        """
        Validate FHIR resource structure
        
        Args:
            resource: FHIR resource to validate
            
        Returns:
            True if valid, False otherwise
        """
        try:
            # Basic validation - ensure resource has required fields
            if hasattr(resource, 'dict'):
                resource_dict = resource.dict()
                return 'resourceType' in resource_dict or 'id' in resource_dict
            return False
        except Exception as e:
            logger.error(f"❌ FHIR validation error: {str(e)}")
            return False

# Example usage and testing functions
def example_integration():
    """Example of how to use the FHIR integration"""
    
    # Initialize FHIR integration
    fhir = FHIRCancerAlphaIntegration()
    
    # Sample patient data
    patient_data = {
        'patient_id': 'patient-12345',
        'mrn': 'MRN001234',
        'first_name': 'John',
        'last_name': 'Doe',
        'gender': 'male',
        'birth_date': '1975-06-15',
        'phone': '555-0123',
        'email': 'john.doe@example.com'
    }
    
    # Sample prediction data
    prediction = GenomicPrediction(
        prediction_id='pred-12345',
        predicted_cancer_type='BRCA',
        confidence_score=0.95,
        class_probabilities={
            'BRCA': 0.95,
            'LUAD': 0.02,
            'COAD': 0.01,
            'PRAD': 0.01,
            'STAD': 0.01,
            'KIRC': 0.00,
            'HNSC': 0.00,
            'LIHC': 0.00
        },
        processing_time_ms=25.5,
        biological_insights=[
            'High methylation levels detected',
            'BRCA1/BRCA2 pathway involvement indicated',
            'High confidence prediction'
        ],
        shap_explanations={
            'top_features': {
                'methylation_feature_1': 0.45,
                'mutation_feature_7': 0.23,
                'clinical_feature_2': 0.18
            }
        },
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    # Create FHIR resources
    patient = fhir.create_patient_resource(patient_data)
    observation = fhir.create_genomic_observation(patient.id, prediction)
    diagnostic_report = fhir.create_diagnostic_report(
        patient.id, prediction, observations=[observation.id]
    )
    
    # Create bundle
    bundle = fhir.bundle_genomic_resources(patient, observation, diagnostic_report)
    
    print("✅ FHIR integration example completed successfully!")
    print(f"Bundle created with {len(bundle['entry'])} resources")
    
    return bundle

if __name__ == "__main__":
    example_integration()
