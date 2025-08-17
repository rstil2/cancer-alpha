#!/usr/bin/env python3
"""
Clinical Workflow API for Oncura
=====================================

Provides comprehensive clinical workflow integration including order management,
provider notifications, clinical decision support, and result distribution.

Author: Oncura Research Team  
Date: August 2025
Version: 1.0.0
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import uuid
import logging
import asyncio
import json
from enum import Enum
import smtplib
import ssl
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from contextlib import asynccontextmanager

from .fhir_integration import FHIRCancerAlphaIntegration, GenomicPrediction
from .emr_connectors import EMRConnectorFactory, EMRCredentials, OrderRecord, PatientRecord

logger = logging.getLogger(__name__)
security = HTTPBearer()

# Enums for workflow states
class OrderStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"

class NotificationChannel(str, Enum):
    EMAIL = "email"
    SMS = "sms"
    EMR_MESSAGE = "emr_message"
    PORTAL_ALERT = "portal_alert"

class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    URGENT = "urgent"

# Pydantic models
class GenomicTestOrder(BaseModel):
    """Genomic test order request"""
    order_id: Optional[str] = Field(None, description="System-generated order ID")
    patient_mrn: str = Field(..., description="Patient medical record number")
    patient_id: Optional[str] = Field(None, description="EMR patient identifier")
    ordered_by_provider: str = Field(..., description="Ordering provider identifier")
    test_type: str = Field("cancer_genomics_classification", description="Type of genomic test")
    priority: str = Field("routine", description="Order priority (routine, urgent, stat)")
    clinical_indication: str = Field(..., description="Clinical reason for testing")
    specimen_type: str = Field("tissue", description="Type of specimen")
    order_notes: Optional[str] = Field(None, description="Additional notes")
    callback_url: Optional[str] = Field(None, description="URL for result notifications")

class GenomicTestResult(BaseModel):
    """Genomic test result"""
    result_id: str = Field(..., description="Unique result identifier")
    order_id: str = Field(..., description="Associated order ID")
    patient_mrn: str = Field(..., description="Patient MRN")
    predicted_cancer_type: str = Field(..., description="Predicted cancer type")
    confidence_score: float = Field(..., ge=0, le=1, description="Prediction confidence")
    class_probabilities: Dict[str, float] = Field(..., description="Probabilities for all cancer types")
    biological_insights: List[str] = Field(..., description="Clinical insights")
    processing_time_ms: float = Field(..., description="Processing time")
    result_date: str = Field(..., description="Result timestamp")
    reviewed_by: Optional[str] = Field(None, description="Provider who reviewed results")
    status: str = Field("preliminary", description="Result status")

class ProviderNotification(BaseModel):
    """Provider notification request"""
    provider_id: str = Field(..., description="Provider identifier")
    patient_mrn: str = Field(..., description="Patient MRN")
    message_type: str = Field(..., description="Type of notification")
    subject: str = Field(..., description="Notification subject")
    content: str = Field(..., description="Notification content")
    channels: List[NotificationChannel] = Field(..., description="Notification channels")
    severity: AlertSeverity = Field(AlertSeverity.INFO, description="Alert severity")
    action_required: bool = Field(False, description="Whether action is required")

class ClinicalDecisionAlert(BaseModel):
    """Clinical decision support alert"""
    alert_id: Optional[str] = Field(None, description="System-generated alert ID")
    patient_mrn: str = Field(..., description="Patient MRN")
    alert_type: str = Field(..., description="Type of clinical alert")
    severity: AlertSeverity = Field(..., description="Alert severity")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Alert description")
    recommendations: List[str] = Field(..., description="Clinical recommendations")
    evidence: Dict[str, Any] = Field(..., description="Supporting evidence")
    expires_at: Optional[str] = Field(None, description="Alert expiration")
    dismissed: bool = Field(False, description="Whether alert was dismissed")

class WorkflowStatus(BaseModel):
    """Overall workflow status"""
    total_orders: int
    pending_orders: int
    processing_orders: int
    completed_orders: int
    failed_orders: int
    average_processing_time: float
    system_health: str

# Global workflow management
workflow_orders: Dict[str, Dict] = {}
active_alerts: Dict[str, Dict] = {}
notification_queue: List[Dict] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("🚀 Starting Clinical Workflow API...")
    # Start background task for processing notifications
    asyncio.create_task(process_notification_queue())
    yield
    # Shutdown
    logger.info("Shutting down Clinical Workflow API...")

app = FastAPI(
    title="Oncura - Clinical Workflow API",
    description="Comprehensive clinical workflow integration for Oncura genomic predictions",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://*.hospital.com", "https://*.clinic.com", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def verify_clinical_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify clinical API key authentication"""
    valid_keys = [
        "clinical-workflow-key-2025",
        "hospital-integration-key",
        "provider-dashboard-key",
        "emr-connector-key"
    ]
    
    if credentials.credentials not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid clinical API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

@app.post("/orders/create", response_model=Dict[str, str])
async def create_genomic_order(
    order: GenomicTestOrder,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_clinical_api_key)
):
    """
    Create a new genomic test order
    
    This endpoint allows healthcare providers to order Oncura genomic
    classification tests through their EMR systems or clinical applications.
    """
    try:
        # Generate order ID if not provided
        if not order.order_id:
            order.order_id = f"CA-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}"
        
        # Create order record
        order_data = {
            "order_id": order.order_id,
            "patient_mrn": order.patient_mrn,
            "patient_id": order.patient_id,
            "ordered_by": order.ordered_by_provider,
            "test_type": order.test_type,
            "priority": order.priority,
            "clinical_indication": order.clinical_indication,
            "specimen_type": order.specimen_type,
            "order_notes": order.order_notes,
            "callback_url": order.callback_url,
            "status": OrderStatus.PENDING,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Store order in workflow system
        workflow_orders[order.order_id] = order_data
        
        # Schedule background processing
        background_tasks.add_task(process_genomic_order, order.order_id)
        
        # Notify ordering provider
        background_tasks.add_task(
            notify_provider_order_received,
            order.ordered_by_provider,
            order.patient_mrn,
            order.order_id
        )
        
        logger.info(f"✅ Created genomic order {order.order_id} for patient {order.patient_mrn}")
        
        return {
            "order_id": order.order_id,
            "status": "created",
            "message": "Genomic test order created successfully",
            "estimated_completion": (datetime.now() + timedelta(hours=2)).isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error creating genomic order: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create genomic order: {str(e)}"
        )

@app.get("/orders/{order_id}", response_model=Dict[str, Any])
async def get_order_status(
    order_id: str,
    api_key: str = Depends(verify_clinical_api_key)
):
    """
    Get status and details of a specific genomic test order
    """
    try:
        if order_id not in workflow_orders:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Order {order_id} not found"
            )
        
        order_data = workflow_orders[order_id]
        
        return {
            "order_id": order_id,
            "status": order_data["status"],
            "patient_mrn": order_data["patient_mrn"],
            "test_type": order_data["test_type"],
            "priority": order_data["priority"],
            "created_at": order_data["created_at"],
            "updated_at": order_data["updated_at"],
            "results_available": order_data.get("results_available", False),
            "estimated_completion": order_data.get("estimated_completion"),
            "processing_notes": order_data.get("processing_notes", [])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error retrieving order status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve order status: {str(e)}"
        )

@app.get("/orders", response_model=List[Dict[str, Any]])
async def get_orders(
    status: Optional[OrderStatus] = None,
    provider_id: Optional[str] = None,
    patient_mrn: Optional[str] = None,
    limit: int = 50,
    api_key: str = Depends(verify_clinical_api_key)
):
    """
    Get list of genomic test orders with optional filtering
    """
    try:
        orders = []
        
        for order_id, order_data in workflow_orders.items():
            # Apply filters
            if status and order_data["status"] != status:
                continue
            if provider_id and order_data["ordered_by"] != provider_id:
                continue  
            if patient_mrn and order_data["patient_mrn"] != patient_mrn:
                continue
                
            orders.append({
                "order_id": order_id,
                "status": order_data["status"],
                "patient_mrn": order_data["patient_mrn"],
                "ordered_by": order_data["ordered_by"],
                "test_type": order_data["test_type"],
                "priority": order_data["priority"],
                "created_at": order_data["created_at"],
                "updated_at": order_data["updated_at"]
            })
            
            if len(orders) >= limit:
                break
        
        # Sort by creation date (most recent first)
        orders.sort(key=lambda x: x["created_at"], reverse=True)
        
        return orders
        
    except Exception as e:
        logger.error(f"❌ Error retrieving orders: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve orders: {str(e)}"
        )

@app.post("/results/submit")
async def submit_genomic_results(
    result: GenomicTestResult,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_clinical_api_key)
):
    """
    Submit genomic test results and trigger clinical workflow
    """
    try:
        # Validate order exists
        if result.order_id not in workflow_orders:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Order {result.order_id} not found"
            )
        
        order_data = workflow_orders[result.order_id]
        
        # Update order with results
        order_data["status"] = OrderStatus.COMPLETED
        order_data["results_available"] = True
        order_data["result_id"] = result.result_id
        order_data["results"] = result.dict()
        order_data["updated_at"] = datetime.now().isoformat()
        
        # Create clinical decision support alerts if needed
        background_tasks.add_task(
            evaluate_clinical_alerts,
            result.patient_mrn,
            result.dict()
        )
        
        # Notify ordering provider
        background_tasks.add_task(
            notify_provider_results_available,
            order_data["ordered_by"],
            result.patient_mrn,
            result.order_id,
            result.dict()
        )
        
        # Post results to EMR if configured
        if order_data.get("callback_url"):
            background_tasks.add_task(
                post_results_to_emr,
                order_data["patient_id"],
                result.dict(),
                order_data.get("emr_type", "epic")
            )
        
        logger.info(f"✅ Submitted results for order {result.order_id}")
        
        return {
            "status": "success",
            "message": "Results submitted successfully",
            "result_id": result.result_id,
            "notifications_sent": True,
            "emr_updated": bool(order_data.get("callback_url"))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error submitting results: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit results: {str(e)}"
        )

@app.post("/notifications/send")
async def send_provider_notification(
    notification: ProviderNotification,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(verify_clinical_api_key)
):
    """
    Send notification to healthcare provider
    """
    try:
        notification_id = str(uuid.uuid4())
        
        # Queue notification for processing
        notification_data = {
            "notification_id": notification_id,
            "provider_id": notification.provider_id,
            "patient_mrn": notification.patient_mrn,
            "message_type": notification.message_type,
            "subject": notification.subject,
            "content": notification.content,
            "channels": notification.channels,
            "severity": notification.severity,
            "action_required": notification.action_required,
            "created_at": datetime.now().isoformat(),
            "status": "queued"
        }
        
        notification_queue.append(notification_data)
        
        logger.info(f"✅ Queued notification {notification_id} for provider {notification.provider_id}")
        
        return {
            "notification_id": notification_id,
            "status": "queued",
            "message": "Notification queued for delivery",
            "channels": notification.channels,
            "estimated_delivery": (datetime.now() + timedelta(minutes=5)).isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Error sending notification: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to send notification: {str(e)}"
        )

@app.post("/alerts/create")
async def create_clinical_alert(
    alert: ClinicalDecisionAlert,
    api_key: str = Depends(verify_clinical_api_key)
):
    """
    Create clinical decision support alert
    """
    try:
        # Generate alert ID if not provided
        if not alert.alert_id:
            alert.alert_id = f"CDS-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}"
        
        alert_data = {
            "alert_id": alert.alert_id,
            "patient_mrn": alert.patient_mrn,
            "alert_type": alert.alert_type,
            "severity": alert.severity,
            "title": alert.title,
            "description": alert.description,
            "recommendations": alert.recommendations,
            "evidence": alert.evidence,
            "expires_at": alert.expires_at,
            "dismissed": alert.dismissed,
            "created_at": datetime.now().isoformat(),
            "active": True
        }
        
        # Store active alert
        active_alerts[alert.alert_id] = alert_data
        
        logger.info(f"✅ Created clinical alert {alert.alert_id} for patient {alert.patient_mrn}")
        
        return {
            "alert_id": alert.alert_id,
            "status": "created",
            "message": "Clinical decision support alert created",
            "severity": alert.severity,
            "active": True
        }
        
    except Exception as e:
        logger.error(f"❌ Error creating clinical alert: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create clinical alert: {str(e)}"
        )

@app.get("/alerts", response_model=List[Dict[str, Any]])
async def get_clinical_alerts(
    patient_mrn: Optional[str] = None,
    severity: Optional[AlertSeverity] = None,
    active_only: bool = True,
    api_key: str = Depends(verify_clinical_api_key)
):
    """
    Get clinical decision support alerts
    """
    try:
        alerts = []
        
        for alert_id, alert_data in active_alerts.items():
            # Apply filters
            if patient_mrn and alert_data["patient_mrn"] != patient_mrn:
                continue
            if severity and alert_data["severity"] != severity:
                continue
            if active_only and not alert_data.get("active", True):
                continue
                
            alerts.append({
                "alert_id": alert_id,
                "patient_mrn": alert_data["patient_mrn"],
                "alert_type": alert_data["alert_type"],
                "severity": alert_data["severity"],
                "title": alert_data["title"],
                "description": alert_data["description"],
                "recommendations": alert_data["recommendations"],
                "created_at": alert_data["created_at"],
                "expires_at": alert_data.get("expires_at"),
                "dismissed": alert_data.get("dismissed", False)
            })
        
        # Sort by severity and creation date
        severity_order = {"urgent": 0, "critical": 1, "warning": 2, "info": 3}
        alerts.sort(key=lambda x: (severity_order.get(x["severity"], 4), x["created_at"]))
        
        return alerts
        
    except Exception as e:
        logger.error(f"❌ Error retrieving alerts: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve alerts: {str(e)}"
        )

@app.get("/workflow/status", response_model=WorkflowStatus)
async def get_workflow_status(api_key: str = Depends(verify_clinical_api_key)):
    """
    Get overall clinical workflow status and metrics
    """
    try:
        total_orders = len(workflow_orders)
        
        status_counts = {
            "pending": 0,
            "in_progress": 0,
            "completed": 0,
            "failed": 0
        }
        
        processing_times = []
        
        for order_data in workflow_orders.values():
            status = order_data.get("status", "pending")
            if status in status_counts:
                status_counts[status] += 1
            
            if status == "completed" and "results" in order_data:
                results = order_data["results"]
                if "processing_time_ms" in results:
                    processing_times.append(results["processing_time_ms"])
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Determine system health
        completion_rate = status_counts["completed"] / max(total_orders, 1)
        failure_rate = status_counts["failed"] / max(total_orders, 1)
        
        if failure_rate > 0.1:
            system_health = "degraded"
        elif completion_rate > 0.8:
            system_health = "healthy"
        else:
            system_health = "operational"
        
        return WorkflowStatus(
            total_orders=total_orders,
            pending_orders=status_counts["pending"],
            processing_orders=status_counts["in_progress"],
            completed_orders=status_counts["completed"],
            failed_orders=status_counts["failed"],
            average_processing_time=avg_processing_time,
            system_health=system_health
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting workflow status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow status: {str(e)}"
        )

# Background task functions
async def process_genomic_order(order_id: str):
    """Process genomic test order in background"""
    try:
        if order_id not in workflow_orders:
            return
        
        order_data = workflow_orders[order_id]
        order_data["status"] = OrderStatus.IN_PROGRESS
        order_data["updated_at"] = datetime.now().isoformat()
        
        # Simulate processing time based on priority
        if order_data["priority"] == "stat":
            await asyncio.sleep(60)  # 1 minute for stat orders
        elif order_data["priority"] == "urgent":
            await asyncio.sleep(300)  # 5 minutes for urgent orders
        else:
            await asyncio.sleep(900)  # 15 minutes for routine orders
        
        # Here we would integrate with the actual Oncura prediction API
        # For now, we'll simulate completion
        order_data["estimated_completion"] = datetime.now().isoformat()
        order_data["processing_notes"] = [
            "Specimen received and processed",
            "Genomic features extracted",
            "Oncura AI analysis complete"
        ]
        
        logger.info(f"✅ Processed genomic order {order_id}")
        
    except Exception as e:
        if order_id in workflow_orders:
            workflow_orders[order_id]["status"] = OrderStatus.FAILED
            workflow_orders[order_id]["error_message"] = str(e)
        logger.error(f"❌ Error processing order {order_id}: {str(e)}")

async def notify_provider_order_received(provider_id: str, patient_mrn: str, order_id: str):
    """Notify provider that order was received"""
    try:
        notification = {
            "provider_id": provider_id,
            "patient_mrn": patient_mrn,
            "subject": f"Genomic Test Order Received - {patient_mrn}",
            "content": f"Your genomic test order {order_id} for patient {patient_mrn} has been received and is being processed.",
            "channels": [NotificationChannel.EMR_MESSAGE, NotificationChannel.EMAIL],
            "severity": AlertSeverity.INFO,
            "created_at": datetime.now().isoformat()
        }
        
        notification_queue.append(notification)
        logger.info(f"✅ Queued order received notification for provider {provider_id}")
        
    except Exception as e:
        logger.error(f"❌ Error notifying provider about order: {str(e)}")

async def notify_provider_results_available(provider_id: str, patient_mrn: str, 
                                          order_id: str, results: Dict[str, Any]):
    """Notify provider that results are available"""
    try:
        # Determine notification urgency based on results
        confidence = results.get("confidence_score", 0)
        cancer_type = results.get("predicted_cancer_type", "Unknown")
        
        if confidence > 0.9:
            severity = AlertSeverity.CRITICAL
            action_required = True
        elif confidence > 0.7:
            severity = AlertSeverity.WARNING
            action_required = True
        else:
            severity = AlertSeverity.INFO
            action_required = False
        
        content = f"""
Oncura genomic analysis results are now available for patient {patient_mrn}.

Key Findings:
- Predicted Cancer Type: {cancer_type}
- Confidence Score: {confidence:.1%}
- Order ID: {order_id}

Please review the complete results in your EMR system or provider dashboard.
        """.strip()
        
        notification = {
            "provider_id": provider_id,
            "patient_mrn": patient_mrn,
            "subject": f"URGENT: Oncura Results Available - {patient_mrn}",
            "content": content,
            "channels": [NotificationChannel.EMR_MESSAGE, NotificationChannel.EMAIL, NotificationChannel.PORTAL_ALERT],
            "severity": severity,
            "action_required": action_required,
            "created_at": datetime.now().isoformat()
        }
        
        notification_queue.append(notification)
        logger.info(f"✅ Queued results notification for provider {provider_id}")
        
    except Exception as e:
        logger.error(f"❌ Error notifying provider about results: {str(e)}")

async def evaluate_clinical_alerts(patient_mrn: str, results: Dict[str, Any]):
    """Evaluate if clinical decision support alerts should be created"""
    try:
        confidence = results.get("confidence_score", 0)
        cancer_type = results.get("predicted_cancer_type", "")
        insights = results.get("biological_insights", [])
        
        alerts_to_create = []
        
        # High confidence prediction alert
        if confidence > 0.95:
            alerts_to_create.append({
                "patient_mrn": patient_mrn,
                "alert_type": "high_confidence_prediction",
                "severity": AlertSeverity.CRITICAL,
                "title": "High Confidence Cancer Prediction",
                "description": f"Oncura has made a high confidence prediction ({confidence:.1%}) for {cancer_type}",
                "recommendations": [
                    "Review complete genomic analysis results",
                    "Consider immediate oncology consultation",
                    "Verify findings with additional testing if needed"
                ],
                "evidence": results
            })
        
        # Specific cancer type alerts
        if cancer_type == "BRCA":
            alerts_to_create.append({
                "patient_mrn": patient_mrn,
                "alert_type": "brca_prediction",
                "severity": AlertSeverity.WARNING,
                "title": "Breast Cancer Signature Detected",
                "description": "Genomic analysis suggests breast cancer signature",
                "recommendations": [
                    "Consider BRCA1/BRCA2 genetic testing",
                    "Evaluate family history",
                    "Schedule breast imaging if not recent"
                ],
                "evidence": results
            })
        
        # Create alerts
        for alert_data in alerts_to_create:
            alert_id = f"CDS-{datetime.now().strftime('%Y%m%d')}-{str(uuid.uuid4())[:8]}"
            alert_data["alert_id"] = alert_id
            alert_data["created_at"] = datetime.now().isoformat()
            alert_data["active"] = True
            alert_data["dismissed"] = False
            
            active_alerts[alert_id] = alert_data
            logger.info(f"✅ Created clinical alert {alert_id} for patient {patient_mrn}")
        
    except Exception as e:
        logger.error(f"❌ Error evaluating clinical alerts: {str(e)}")

async def post_results_to_emr(patient_id: str, results: Dict[str, Any], emr_type: str):
    """Post results back to EMR system"""
    try:
        # This would integrate with the EMR connector
        # For demonstration, we'll just log the action
        logger.info(f"✅ Would post results to {emr_type} EMR for patient {patient_id}")
        
        # In a real implementation:
        # emr_connector = EMRConnectorFactory.create_connector(emr_type, credentials)
        # await emr_connector.post_results(patient_id, results)
        
    except Exception as e:
        logger.error(f"❌ Error posting results to EMR: {str(e)}")

async def process_notification_queue():
    """Background task to process notification queue"""
    while True:
        try:
            if notification_queue:
                notification = notification_queue.pop(0)
                await send_notification(notification)
            
            await asyncio.sleep(10)  # Check every 10 seconds
            
        except Exception as e:
            logger.error(f"❌ Error processing notification queue: {str(e)}")
            await asyncio.sleep(30)  # Wait longer on error

async def send_notification(notification: Dict[str, Any]):
    """Send individual notification"""
    try:
        channels = notification.get("channels", [])
        
        for channel in channels:
            if channel == NotificationChannel.EMAIL:
                await send_email_notification(notification)
            elif channel == NotificationChannel.EMR_MESSAGE:
                await send_emr_message(notification)
            elif channel == NotificationChannel.PORTAL_ALERT:
                await send_portal_alert(notification)
        
        logger.info(f"✅ Sent notification via {len(channels)} channels")
        
    except Exception as e:
        logger.error(f"❌ Error sending notification: {str(e)}")

async def send_email_notification(notification: Dict[str, Any]):
    """Send email notification"""
    try:
        # In a real implementation, this would send actual emails
        logger.info(f"📧 Would send email to provider {notification['provider_id']}")
        logger.info(f"   Subject: {notification['subject']}")
        
    except Exception as e:
        logger.error(f"❌ Error sending email: {str(e)}")

async def send_emr_message(notification: Dict[str, Any]):
    """Send EMR message"""
    try:
        # In a real implementation, this would use EMR messaging APIs
        logger.info(f"💬 Would send EMR message to provider {notification['provider_id']}")
        
    except Exception as e:
        logger.error(f"❌ Error sending EMR message: {str(e)}")

async def send_portal_alert(notification: Dict[str, Any]):
    """Send portal alert"""
    try:
        # In a real implementation, this would create portal alerts
        logger.info(f"🔔 Would create portal alert for provider {notification['provider_id']}")
        
    except Exception as e:
        logger.error(f"❌ Error sending portal alert: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Oncura - Clinical Workflow API",
        "version": "1.0.0",
        "description": "Comprehensive clinical workflow integration for genomic predictions",
        "endpoints": {
            "orders": "Order management and tracking",
            "results": "Result submission and distribution",
            "notifications": "Provider notification system",
            "alerts": "Clinical decision support alerts",
            "workflow": "Workflow status and metrics"
        },
        "status": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "clinical_workflow_api:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )
