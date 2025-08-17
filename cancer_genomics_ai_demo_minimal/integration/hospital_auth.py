#!/usr/bin/env python3
"""
Hospital Authentication System for Cancer Alpha
===============================================

Provides comprehensive authentication and authorization for hospital environments
including SSO integration, LDAP/Active Directory support, and role-based access control.

Author: Cancer Alpha Research Team
Date: August 2025
Version: 1.0.0
"""

import jwt
import ldap
import ssl
import hashlib
import secrets
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from fastapi import FastAPI, HTTPException, Depends, Security, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import requests
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
import redis
import asyncio
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

# Security schemes
security = HTTPBearer()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# User roles and permissions
class UserRole(str, Enum):
    ADMIN = "admin"
    PHYSICIAN = "physician"
    NURSE = "nurse"
    LAB_TECH = "lab_tech"
    RESEARCHER = "researcher"
    VIEWER = "viewer"

class Permission(str, Enum):
    READ_PATIENT_DATA = "read_patient_data"
    WRITE_PATIENT_DATA = "write_patient_data"
    ORDER_TESTS = "order_tests"
    VIEW_RESULTS = "view_results"
    APPROVE_RESULTS = "approve_results"
    MANAGE_USERS = "manage_users"
    VIEW_REPORTS = "view_reports"
    EXPORT_DATA = "export_data"

# Role-based permissions mapping
ROLE_PERMISSIONS = {
    UserRole.ADMIN: [
        Permission.READ_PATIENT_DATA,
        Permission.WRITE_PATIENT_DATA,
        Permission.ORDER_TESTS,
        Permission.VIEW_RESULTS,
        Permission.APPROVE_RESULTS,
        Permission.MANAGE_USERS,
        Permission.VIEW_REPORTS,
        Permission.EXPORT_DATA
    ],
    UserRole.PHYSICIAN: [
        Permission.READ_PATIENT_DATA,
        Permission.WRITE_PATIENT_DATA,
        Permission.ORDER_TESTS,
        Permission.VIEW_RESULTS,
        Permission.APPROVE_RESULTS,
        Permission.VIEW_REPORTS
    ],
    UserRole.NURSE: [
        Permission.READ_PATIENT_DATA,
        Permission.VIEW_RESULTS,
        Permission.VIEW_REPORTS
    ],
    UserRole.LAB_TECH: [
        Permission.READ_PATIENT_DATA,
        Permission.VIEW_RESULTS,
        Permission.ORDER_TESTS
    ],
    UserRole.RESEARCHER: [
        Permission.VIEW_RESULTS,
        Permission.VIEW_REPORTS,
        Permission.EXPORT_DATA
    ],
    UserRole.VIEWER: [
        Permission.VIEW_RESULTS,
        Permission.VIEW_REPORTS
    ]
}

# Pydantic models
@dataclass
class HospitalUser:
    """Hospital user data structure"""
    user_id: str
    username: str
    email: str
    first_name: str
    last_name: str
    role: UserRole
    department: str
    hospital_id: str
    npi_number: Optional[str] = None
    active: bool = True
    last_login: Optional[str] = None
    permissions: Optional[List[Permission]] = None

class LoginRequest(BaseModel):
    """User login request"""
    username: str = Field(..., description="Hospital username or employee ID")
    password: str = Field(..., description="User password")
    hospital_domain: Optional[str] = Field(None, description="Hospital domain for multi-tenant")

class SSOLoginRequest(BaseModel):
    """SSO login request"""
    sso_token: str = Field(..., description="SSO token from identity provider")
    provider: str = Field(..., description="SSO provider (saml, oidc, oauth2)")
    hospital_domain: Optional[str] = Field(None, description="Hospital domain")

class TokenResponse(BaseModel):
    """Authentication token response"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    refresh_token: str
    user_info: Dict[str, Any]

class UserInfo(BaseModel):
    """User information response"""
    user_id: str
    username: str
    email: str
    first_name: str
    last_name: str
    role: str
    department: str
    hospital_id: str
    permissions: List[str]
    last_login: Optional[str]

# Authentication configuration
@dataclass
class AuthConfig:
    """Authentication configuration"""
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 60
    refresh_token_expire_days: int = 7
    ldap_server: Optional[str] = None
    ldap_port: int = 389
    ldap_use_ssl: bool = False
    ldap_base_dn: str = ""
    ldap_user_dn_template: str = ""
    ldap_search_filter: str = "(uid={username})"
    sso_enabled: bool = False
    saml_metadata_url: Optional[str] = None
    oidc_discovery_url: Optional[str] = None
    oauth2_client_id: Optional[str] = None
    oauth2_client_secret: Optional[str] = None

class HospitalAuthenticationSystem:
    """
    Comprehensive hospital authentication system
    
    Features:
    - Username/password authentication
    - LDAP/Active Directory integration
    - SAML 2.0 SSO support
    - OpenID Connect support
    - OAuth 2.0 integration
    - Role-based access control
    - Multi-tenant hospital support
    - Session management
    - Audit logging
    """
    
    def __init__(self, config: AuthConfig):
        self.config = config
        self.redis_client = None
        self.active_sessions: Dict[str, Dict] = {}
        self.user_cache: Dict[str, HospitalUser] = {}
        
        # Initialize Redis for session management
        try:
            self.redis_client = redis.Redis(
                host='localhost', 
                port=6379, 
                decode_responses=True,
                socket_timeout=1
            )
            self.redis_client.ping()
            logger.info("✅ Redis connected for session management")
        except:
            logger.warning("⚠️ Redis not available, using in-memory session storage")
    
    async def authenticate_user(self, username: str, password: str, 
                              hospital_domain: Optional[str] = None) -> Optional[HospitalUser]:
        """
        Authenticate user with username and password
        
        Args:
            username: Hospital username or employee ID
            password: User password
            hospital_domain: Hospital domain for multi-tenant setups
            
        Returns:
            HospitalUser object if authentication successful, None otherwise
        """
        try:
            # Try LDAP authentication first if configured
            if self.config.ldap_server:
                user = await self._authenticate_ldap(username, password, hospital_domain)
                if user:
                    return user
            
            # Fall back to local authentication
            user = await self._authenticate_local(username, password, hospital_domain)
            return user
            
        except Exception as e:
            logger.error(f"❌ Authentication error for user {username}: {str(e)}")
            return None
    
    async def authenticate_sso(self, sso_token: str, provider: str, 
                             hospital_domain: Optional[str] = None) -> Optional[HospitalUser]:
        """
        Authenticate user with SSO token
        
        Args:
            sso_token: SSO token from identity provider
            provider: SSO provider type (saml, oidc, oauth2)
            hospital_domain: Hospital domain
            
        Returns:
            HospitalUser object if authentication successful, None otherwise
        """
        try:
            if provider.lower() == "saml":
                return await self._authenticate_saml(sso_token, hospital_domain)
            elif provider.lower() == "oidc":
                return await self._authenticate_oidc(sso_token, hospital_domain)
            elif provider.lower() == "oauth2":
                return await self._authenticate_oauth2(sso_token, hospital_domain)
            else:
                logger.error(f"❌ Unsupported SSO provider: {provider}")
                return None
                
        except Exception as e:
            logger.error(f"❌ SSO authentication error: {str(e)}")
            return None
    
    async def _authenticate_ldap(self, username: str, password: str, 
                               hospital_domain: Optional[str] = None) -> Optional[HospitalUser]:
        """Authenticate against LDAP/Active Directory"""
        try:
            # Configure LDAP connection
            ldap_url = f"ldap{'s' if self.config.ldap_use_ssl else ''}://{self.config.ldap_server}:{self.config.ldap_port}"
            
            # Initialize LDAP connection
            ldap_conn = ldap.initialize(ldap_url)
            ldap_conn.protocol_version = ldap.VERSION3
            ldap_conn.set_option(ldap.OPT_REFERRALS, 0)
            
            if self.config.ldap_use_ssl:
                ldap_conn.set_option(ldap.OPT_X_TLS_REQUIRE_CERT, ldap.OPT_X_TLS_NEVER)
                ldap_conn.start_tls_s()
            
            # Construct user DN
            if "{username}" in self.config.ldap_user_dn_template:
                user_dn = self.config.ldap_user_dn_template.format(username=username)
            else:
                user_dn = f"uid={username},{self.config.ldap_base_dn}"
            
            # Attempt to bind with user credentials
            ldap_conn.simple_bind_s(user_dn, password)
            
            # Search for user attributes
            search_filter = self.config.ldap_search_filter.format(username=username)
            result = ldap_conn.search_s(
                self.config.ldap_base_dn,
                ldap.SCOPE_SUBTREE,
                search_filter,
                ['cn', 'mail', 'givenName', 'sn', 'employeeID', 'department', 'title']
            )
            
            if result:
                dn, attrs = result[0]
                
                # Extract user information
                user = HospitalUser(
                    user_id=attrs.get('employeeID', [username.encode()])[0].decode(),
                    username=username,
                    email=attrs.get('mail', [b''])[0].decode(),
                    first_name=attrs.get('givenName', [b''])[0].decode(),
                    last_name=attrs.get('sn', [b''])[0].decode(),
                    role=self._map_ldap_role_to_user_role(attrs.get('title', [b''])[0].decode()),
                    department=attrs.get('department', [b''])[0].decode(),
                    hospital_id=hospital_domain or 'default',
                    active=True,
                    last_login=datetime.now().isoformat()
                )
                
                # Set permissions based on role
                user.permissions = ROLE_PERMISSIONS.get(user.role, [])
                
                # Cache user information
                self.user_cache[user.user_id] = user
                
                logger.info(f"✅ LDAP authentication successful for user {username}")
                return user
            
        except ldap.INVALID_CREDENTIALS:
            logger.warning(f"⚠️ Invalid LDAP credentials for user {username}")
        except Exception as e:
            logger.error(f"❌ LDAP authentication error: {str(e)}")
        finally:
            try:
                ldap_conn.unbind_s()
            except:
                pass
        
        return None
    
    async def _authenticate_local(self, username: str, password: str, 
                                hospital_domain: Optional[str] = None) -> Optional[HospitalUser]:
        """Authenticate against local user database"""
        try:
            # In a real implementation, this would query a database
            # For demonstration, we'll use hardcoded users
            local_users = {
                "doctor1": {
                    "password_hash": self._hash_password("password123"),
                    "user_data": {
                        "user_id": "DOC001",
                        "username": "doctor1",
                        "email": "doctor1@hospital.com",
                        "first_name": "John",
                        "last_name": "Doe",
                        "role": UserRole.PHYSICIAN,
                        "department": "Oncology",
                        "hospital_id": hospital_domain or "main_hospital",
                        "npi_number": "1234567890"
                    }
                },
                "nurse1": {
                    "password_hash": self._hash_password("nurse123"),
                    "user_data": {
                        "user_id": "NUR001",
                        "username": "nurse1",
                        "email": "nurse1@hospital.com",
                        "first_name": "Jane",
                        "last_name": "Smith",
                        "role": UserRole.NURSE,
                        "department": "Oncology",
                        "hospital_id": hospital_domain or "main_hospital"
                    }
                },
                "admin": {
                    "password_hash": self._hash_password("admin123"),
                    "user_data": {
                        "user_id": "ADM001",
                        "username": "admin",
                        "email": "admin@hospital.com",
                        "first_name": "Cancer Alpha",
                        "last_name": "Administrator",
                        "role": UserRole.ADMIN,
                        "department": "IT",
                        "hospital_id": hospital_domain or "main_hospital"
                    }
                }
            }
            
            if username in local_users:
                stored_hash = local_users[username]["password_hash"]
                if self._verify_password(password, stored_hash):
                    user_data = local_users[username]["user_data"]
                    user = HospitalUser(
                        **user_data,
                        active=True,
                        last_login=datetime.now().isoformat()
                    )
                    
                    # Set permissions based on role
                    user.permissions = ROLE_PERMISSIONS.get(user.role, [])
                    
                    # Cache user information
                    self.user_cache[user.user_id] = user
                    
                    logger.info(f"✅ Local authentication successful for user {username}")
                    return user
            
            logger.warning(f"⚠️ Invalid local credentials for user {username}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Local authentication error: {str(e)}")
            return None
    
    async def _authenticate_saml(self, sso_token: str, hospital_domain: Optional[str] = None) -> Optional[HospitalUser]:
        """Authenticate SAML SSO token"""
        try:
            # In a real implementation, this would:
            # 1. Validate SAML assertion signature
            # 2. Check assertion conditions (time bounds, audience, etc.)
            # 3. Extract user attributes from assertion
            # 4. Map to internal user structure
            
            logger.info("🔐 Processing SAML SSO authentication")
            
            # Mock SAML processing for demonstration
            # In production, use libraries like python3-saml
            decoded_token = jwt.decode(sso_token, options={"verify_signature": False})
            
            user = HospitalUser(
                user_id=decoded_token.get("sub", "saml_user"),
                username=decoded_token.get("preferred_username", "saml_user"),
                email=decoded_token.get("email", ""),
                first_name=decoded_token.get("given_name", ""),
                last_name=decoded_token.get("family_name", ""),
                role=UserRole.PHYSICIAN,  # Would be mapped from SAML attributes
                department=decoded_token.get("department", ""),
                hospital_id=hospital_domain or "saml_hospital",
                active=True,
                last_login=datetime.now().isoformat()
            )
            
            user.permissions = ROLE_PERMISSIONS.get(user.role, [])
            self.user_cache[user.user_id] = user
            
            logger.info(f"✅ SAML authentication successful for user {user.username}")
            return user
            
        except Exception as e:
            logger.error(f"❌ SAML authentication error: {str(e)}")
            return None
    
    async def _authenticate_oidc(self, sso_token: str, hospital_domain: Optional[str] = None) -> Optional[HospitalUser]:
        """Authenticate OpenID Connect token"""
        try:
            # In a real implementation, this would:
            # 1. Validate JWT signature against OIDC provider's public keys
            # 2. Verify token claims (iss, aud, exp, etc.)
            # 3. Extract user info from token or call userinfo endpoint
            
            logger.info("🔐 Processing OIDC SSO authentication")
            
            # Mock OIDC processing
            decoded_token = jwt.decode(sso_token, options={"verify_signature": False})
            
            user = HospitalUser(
                user_id=decoded_token.get("sub", "oidc_user"),
                username=decoded_token.get("preferred_username", "oidc_user"),
                email=decoded_token.get("email", ""),
                first_name=decoded_token.get("given_name", ""),
                last_name=decoded_token.get("family_name", ""),
                role=UserRole.PHYSICIAN,  # Would be mapped from OIDC claims
                department=decoded_token.get("department", ""),
                hospital_id=hospital_domain or "oidc_hospital",
                active=True,
                last_login=datetime.now().isoformat()
            )
            
            user.permissions = ROLE_PERMISSIONS.get(user.role, [])
            self.user_cache[user.user_id] = user
            
            logger.info(f"✅ OIDC authentication successful for user {user.username}")
            return user
            
        except Exception as e:
            logger.error(f"❌ OIDC authentication error: {str(e)}")
            return None
    
    async def _authenticate_oauth2(self, sso_token: str, hospital_domain: Optional[str] = None) -> Optional[HospitalUser]:
        """Authenticate OAuth 2.0 token"""
        try:
            # In a real implementation, this would:
            # 1. Validate access token with OAuth provider
            # 2. Call user info endpoint to get user details
            # 3. Map provider user info to internal user structure
            
            logger.info("🔐 Processing OAuth2 SSO authentication")
            
            # Mock OAuth2 processing
            if self.config.oidc_discovery_url:
                # Validate token with OAuth provider
                headers = {"Authorization": f"Bearer {sso_token}"}
                response = requests.get(f"{self.config.oidc_discovery_url}/userinfo", headers=headers)
                
                if response.status_code == 200:
                    user_info = response.json()
                    
                    user = HospitalUser(
                        user_id=user_info.get("sub", "oauth_user"),
                        username=user_info.get("preferred_username", "oauth_user"),
                        email=user_info.get("email", ""),
                        first_name=user_info.get("given_name", ""),
                        last_name=user_info.get("family_name", ""),
                        role=UserRole.PHYSICIAN,  # Would be mapped from OAuth claims
                        department=user_info.get("department", ""),
                        hospital_id=hospital_domain or "oauth_hospital",
                        active=True,
                        last_login=datetime.now().isoformat()
                    )
                    
                    user.permissions = ROLE_PERMISSIONS.get(user.role, [])
                    self.user_cache[user.user_id] = user
                    
                    logger.info(f"✅ OAuth2 authentication successful for user {user.username}")
                    return user
            
            return None
            
        except Exception as e:
            logger.error(f"❌ OAuth2 authentication error: {str(e)}")
            return None
    
    def create_access_token(self, user: HospitalUser) -> str:
        """Create JWT access token for user"""
        try:
            now = datetime.utcnow()
            payload = {
                "sub": user.user_id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "hospital_id": user.hospital_id,
                "permissions": [p.value for p in user.permissions] if user.permissions else [],
                "iat": now,
                "exp": now + timedelta(minutes=self.config.access_token_expire_minutes),
                "type": "access_token"
            }
            
            token = jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
            logger.info(f"✅ Created access token for user {user.username}")
            return token
            
        except Exception as e:
            logger.error(f"❌ Error creating access token: {str(e)}")
            raise
    
    def create_refresh_token(self, user: HospitalUser) -> str:
        """Create refresh token for user"""
        try:
            now = datetime.utcnow()
            payload = {
                "sub": user.user_id,
                "username": user.username,
                "hospital_id": user.hospital_id,
                "iat": now,
                "exp": now + timedelta(days=self.config.refresh_token_expire_days),
                "type": "refresh_token"
            }
            
            token = jwt.encode(payload, self.config.jwt_secret, algorithm=self.config.jwt_algorithm)
            logger.info(f"✅ Created refresh token for user {user.username}")
            return token
            
        except Exception as e:
            logger.error(f"❌ Error creating refresh token: {str(e)}")
            raise
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret,
                algorithms=[self.config.jwt_algorithm]
            )
            
            # Check if token is not expired
            if payload.get("exp", 0) < time.time():
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("⚠️ Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("⚠️ Invalid token")
            return None
        except Exception as e:
            logger.error(f"❌ Token verification error: {str(e)}")
            return None
    
    def has_permission(self, user: HospitalUser, required_permission: Permission) -> bool:
        """Check if user has required permission"""
        if not user.permissions:
            return False
        return required_permission in user.permissions
    
    def _map_ldap_role_to_user_role(self, ldap_title: str) -> UserRole:
        """Map LDAP title/role to internal UserRole"""
        title_lower = ldap_title.lower()
        
        if any(term in title_lower for term in ["doctor", "physician", "md", "attending"]):
            return UserRole.PHYSICIAN
        elif "nurse" in title_lower:
            return UserRole.NURSE
        elif any(term in title_lower for term in ["admin", "administrator"]):
            return UserRole.ADMIN
        elif any(term in title_lower for term in ["lab", "technician", "tech"]):
            return UserRole.LAB_TECH
        elif "researcher" in title_lower:
            return UserRole.RESEARCHER
        else:
            return UserRole.VIEWER
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return salt + pwd_hash.hex()
    
    def _verify_password(self, password: str, stored_hash: str) -> bool:
        """Verify password against stored hash"""
        try:
            salt = stored_hash[:32]
            stored_pwd_hash = stored_hash[32:]
            pwd_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
            return pwd_hash.hex() == stored_pwd_hash
        except Exception:
            return False

# FastAPI application setup
auth_config = AuthConfig(
    jwt_secret="cancer-alpha-hospital-auth-secret-2025",
    ldap_server="ldap.hospital.com",
    ldap_base_dn="ou=users,dc=hospital,dc=com",
    ldap_user_dn_template="uid={username},ou=users,dc=hospital,dc=com",
    sso_enabled=True
)

auth_system = HospitalAuthenticationSystem(auth_config)

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Starting Hospital Authentication System...")
    yield
    logger.info("Shutting down Hospital Authentication System...")

app = FastAPI(
    title="Cancer Alpha - Hospital Authentication System",
    description="Comprehensive authentication and authorization for hospital environments",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://*.hospital.com", "https://*.clinic.com", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency for extracting current user from token
async def get_current_user(token: str = Depends(oauth2_scheme)) -> HospitalUser:
    """Extract current user from JWT token"""
    try:
        payload = auth_system.verify_token(token)
        if not payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        user_id = payload.get("sub")
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Try to get user from cache first
        if user_id in auth_system.user_cache:
            return auth_system.user_cache[user_id]
        
        # If not in cache, reconstruct from token
        user = HospitalUser(
            user_id=payload.get("sub"),
            username=payload.get("username"),
            email=payload.get("email"),
            first_name="",  # Not stored in token
            last_name="",   # Not stored in token
            role=UserRole(payload.get("role")),
            department="",  # Not stored in token
            hospital_id=payload.get("hospital_id"),
            permissions=[Permission(p) for p in payload.get("permissions", [])]
        )
        
        return user
        
    except Exception as e:
        logger.error(f"❌ Error extracting user from token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication",
            headers={"WWW-Authenticate": "Bearer"},
        )

def require_permission(required_permission: Permission):
    """Dependency factory for permission checking"""
    def check_permission(current_user: HospitalUser = Depends(get_current_user)):
        if not auth_system.has_permission(current_user, required_permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Insufficient permissions. Required: {required_permission.value}"
            )
        return current_user
    return check_permission

# Authentication endpoints
@app.post("/login", response_model=TokenResponse)
async def login(login_request: LoginRequest):
    """Authenticate user with username and password"""
    try:
        user = await auth_system.authenticate_user(
            login_request.username,
            login_request.password,
            login_request.hospital_domain
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        access_token = auth_system.create_access_token(user)
        refresh_token = auth_system.create_refresh_token(user)
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=auth_config.access_token_expire_minutes * 60,
            refresh_token=refresh_token,
            user_info={
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "department": user.department,
                "hospital_id": user.hospital_id,
                "permissions": [p.value for p in user.permissions] if user.permissions else []
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication system error"
        )

@app.post("/sso/login", response_model=TokenResponse)
async def sso_login(sso_request: SSOLoginRequest):
    """Authenticate user with SSO token"""
    try:
        user = await auth_system.authenticate_sso(
            sso_request.sso_token,
            sso_request.provider,
            sso_request.hospital_domain
        )
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid SSO token",
            )
        
        access_token = auth_system.create_access_token(user)
        refresh_token = auth_system.create_refresh_token(user)
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=auth_config.access_token_expire_minutes * 60,
            refresh_token=refresh_token,
            user_info={
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "department": user.department,
                "hospital_id": user.hospital_id,
                "permissions": [p.value for p in user.permissions] if user.permissions else []
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ SSO login error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="SSO authentication system error"
        )

@app.get("/user/me", response_model=UserInfo)
async def get_current_user_info(current_user: HospitalUser = Depends(get_current_user)):
    """Get current user information"""
    return UserInfo(
        user_id=current_user.user_id,
        username=current_user.username,
        email=current_user.email,
        first_name=current_user.first_name,
        last_name=current_user.last_name,
        role=current_user.role.value,
        department=current_user.department,
        hospital_id=current_user.hospital_id,
        permissions=[p.value for p in current_user.permissions] if current_user.permissions else [],
        last_login=current_user.last_login
    )

@app.post("/token/refresh", response_model=TokenResponse)
async def refresh_token(refresh_token: str):
    """Refresh access token using refresh token"""
    try:
        payload = auth_system.verify_token(refresh_token)
        
        if not payload or payload.get("type") != "refresh_token":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        user_id = payload.get("sub")
        if user_id in auth_system.user_cache:
            user = auth_system.user_cache[user_id]
            
            new_access_token = auth_system.create_access_token(user)
            new_refresh_token = auth_system.create_refresh_token(user)
            
            return TokenResponse(
                access_token=new_access_token,
                token_type="bearer",
                expires_in=auth_config.access_token_expire_minutes * 60,
                refresh_token=new_refresh_token,
                user_info={
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "role": user.role.value,
                    "department": user.department,
                    "hospital_id": user.hospital_id,
                    "permissions": [p.value for p in user.permissions] if user.permissions else []
                }
            )
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Token refresh error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh error"
        )

@app.get("/permissions/check")
async def check_permissions(
    permission: str,
    current_user: HospitalUser = Depends(get_current_user)
):
    """Check if current user has specific permission"""
    try:
        required_permission = Permission(permission)
        has_permission = auth_system.has_permission(current_user, required_permission)
        
        return {
            "user_id": current_user.user_id,
            "permission": permission,
            "has_permission": has_permission,
            "user_role": current_user.role.value,
            "all_permissions": [p.value for p in current_user.permissions] if current_user.permissions else []
        }
        
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid permission: {permission}"
        )

# Protected endpoint examples
@app.get("/admin/users")
async def list_users(current_user: HospitalUser = Depends(require_permission(Permission.MANAGE_USERS))):
    """List all users (admin only)"""
    return {
        "message": "User management endpoint",
        "requested_by": current_user.username,
        "users": list(auth_system.user_cache.keys())
    }

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "name": "Cancer Alpha - Hospital Authentication System",
        "version": "1.0.0",
        "description": "Comprehensive authentication and authorization for hospital environments",
        "features": [
            "Username/password authentication",
            "LDAP/Active Directory integration", 
            "SAML 2.0 SSO support",
            "OpenID Connect support",
            "OAuth 2.0 integration",
            "Role-based access control",
            "Multi-tenant hospital support"
        ],
        "endpoints": {
            "login": "POST /login - Username/password authentication",
            "sso_login": "POST /sso/login - SSO authentication",
            "user_info": "GET /user/me - Current user information",
            "refresh": "POST /token/refresh - Refresh access token",
            "permissions": "GET /permissions/check - Check user permissions"
        },
        "status": "operational"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "hospital_auth:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )
