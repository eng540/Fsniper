"""
Elite Sniper v2.0 - Enhanced Session State Management
Combines multi-session architecture with KingSniperV12 State Machine
Includes Incident System and comprehensive health tracking
"""

import time
import datetime
import uuid
import json
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

logger = logging.getLogger("EliteSniperV2.SessionState")


# ==================== State Machine (from KingSniperV12) ====================
class SystemState(Enum):
    """Overall system operational state"""
    ACTIVE = "ACTIVE"           # Normal operation
    STANDBY = "STANDBY"         # Waiting for scheduled time
    RECOVERING = "RECOVERING"   # In recovery process
    ATTACK = "ATTACK"           # Attack mode (2:00 AM window)


class SessionHealth(Enum):
    """Session health status"""
    CLEAN = "CLEAN"         # Fresh, healthy session
    WARNING = "WARNING"     # Minor issues detected
    DEGRADED = "DEGRADED"   # Significant issues, near expiration
    POISONED = "POISONED"   # Invalid session, must terminate


class SessionRole(Enum):
    """Session role in multi-session architecture"""
    SCOUT = "SCOUT"         # Discovery mode (finds appointments)
    ATTACKER = "ATTACKER"   # Booking mode (executes bookings)


# ==================== Incident Types ====================
class IncidentType(Enum):
    """Types of trackable incidents"""
    CAPTCHA_FAIL = "CAPTCHA_FAIL"
    CAPTCHA_BLACK = "CAPTCHA_BLACK"
    SESSION_EXPIRED = "SESSION_EXPIRED"
    SESSION_POISONED = "SESSION_POISONED"
    DOUBLE_CAPTCHA = "DOUBLE_CAPTCHA"
    NAVIGATION_ERROR = "NAVIGATION_ERROR"
    FORM_REJECTED = "FORM_REJECTED"
    SLOT_DETECTED = "SLOT_DETECTED"
    BOOKING_ATTEMPT = "BOOKING_ATTEMPT"
    BOOKING_SUCCESS = "BOOKING_SUCCESS"
    REBIRTH = "REBIRTH"


class IncidentSeverity(Enum):
    """Incident severity levels"""
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# ==================== Session State ====================
@dataclass
class SessionState:
    """
    Enhanced session state tracking
    Combines Elite Sniper lifecycle management with KingSniperV12 health system
    """
    # Identity
    session_id: str
    role: SessionRole
    worker_id: int
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    start_time: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    last_captcha_time: float = 0.0
    last_captcha_solve_time: float = 0.0  # For double-captcha detection
    attack_mode_entered: float = 0.0       # When attack mode started
    pre_attack_reset_done: bool = False    # Pre-attack reset flag
    
    # Health & Status
    health: SessionHealth = SessionHealth.CLEAN
    captcha_solved: bool = False
    captcha_solve_count: int = 0
    in_captcha_flow: bool = False  # Track if we're in a captcha solve flow
    
    # Counters
    failures: int = 0
    consecutive_errors: int = 0
    consecutive_network_failures: int = 0  # NEW: Track timeouts/network errors
    pages_loaded: int = 0
    navigation_errors: int = 0
    
    # Limits (configurable)
    max_age: float = 60.0           # Maximum session age in seconds
    max_idle: float = 15.0          # Maximum idle time in seconds
    max_failures: int = 3            # Maximum consecutive failures
    max_captcha_attempts: int = 5    # Maximum captcha attempts per session
    
    # Metadata
    current_url: Optional[str] = None
    last_error: Optional[str] = None
    
    def age(self) -> float:
        """Get session age in seconds"""
        return time.time() - self.created_at
    
    def idle_time(self) -> float:
        """Get idle time since last activity"""
        return time.time() - self.last_activity
    
    def touch(self):
        """Update last activity timestamp"""
        self.last_activity = time.time()
        self.consecutive_errors = 0  # Reset on activity
    
    def is_expired(self) -> bool:
        """
        Check if session is expired based on:
        1. Age exceeds max_age
        2. Idle time exceeds max_idle
        """
        age_expired = self.age() > self.max_age
        idle_expired = self.idle_time() > self.max_idle
        
        return age_expired or idle_expired
    
    def mark_captcha_solved(self):
        """Mark captcha as solved and track timing for double-captcha detection"""
        self.captcha_solved = True
        self.captcha_solve_count += 1
        self.last_captcha_solve_time = time.time()
        self.last_captcha_time = time.time()
        self.in_captcha_flow = False
        self.touch()
    
    def is_double_captcha(self) -> bool:
        """
        Check for double-captcha condition (session poisoning indicator)
        Returns True if a captcha was solved recently and another appears
        """
        if not self.captcha_solved:
            return False
        
        # If captcha was solved in the last 30 seconds and we see another
        time_since_last = time.time() - self.last_captcha_solve_time
        return time_since_last < 30.0
    
    def start_captcha_flow(self):
        """Mark start of captcha solving flow"""
        self.in_captcha_flow = True
        self.touch()
    
    def increment_failure(self, error: str = "Unknown"):
        """Increment failure counter"""
        self.failures += 1
        self.consecutive_errors += 1
        self.last_error = error
        
        # Update health based on failures
        if self.consecutive_errors >= self.max_failures:
            self.health = SessionHealth.POISONED
        elif self.consecutive_errors >= 2:
            self.health = SessionHealth.DEGRADED
        elif self.consecutive_errors >= 1:
            self.health = SessionHealth.WARNING
    
    def should_terminate(self) -> bool:
        """
        Determine if session should be terminated immediately
        Based on Elite Sniper kill rules
        """
        # Rule 1: Session expired
        if self.is_expired():
            return True
        
        # Rule 2: Too many failures
        if self.failures >= self.max_failures:
            return True
        
        # Rule 3: Session poisoned
        if self.health == SessionHealth.POISONED:
            return True
        
        # Rule 4: Too many captcha attempts
        if self.captcha_solve_count > self.max_captcha_attempts:
            return True
        
        return False
    
    def reset_for_new_flow(self):
        """Reset session state for new navigation flow"""
        self.captcha_solved = False
        self.in_captcha_flow = False
        self.consecutive_errors = 0
        self.touch()
    
    def enter_attack_mode(self):
        """Mark entry into attack mode"""
        self.attack_mode_entered = time.time()
        self.touch()
    
    def get_health_status(self) -> str:
        """Get human-readable health status"""
        age_str = f"{self.age():.1f}s"
        idle_str = f"{self.idle_time():.1f}s"
        
        return (
            f"Health: {self.health.value} | "
            f"Age: {age_str} | "
            f"Idle: {idle_str} | "
            f"Failures: {self.failures} | "
            f"Captchas: {self.captcha_solve_count}"
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "session_id": self.session_id,
            "role": self.role.value,
            "worker_id": self.worker_id,
            "health": self.health.value,
            "age": self.age(),
            "idle_time": self.idle_time(),
            "failures": self.failures,
            "captcha_solve_count": self.captcha_solve_count,
            "is_expired": self.is_expired(),
            "should_terminate": self.should_terminate()
        }


# ==================== Incident Tracking (from KingSniperV12) ====================
@dataclass
class Incident:
    """Track significant events for debugging and analysis"""
    id: str
    timestamp: datetime.datetime
    session_id: str
    type: IncidentType
    severity: IncidentSeverity
    evidence: Dict[str, Any]
    description: str
    resolved: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging"""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "session_id": self.session_id,
            "type": self.type.value,
            "severity": self.severity.value,
            "evidence": self.evidence,
            "description": self.description,
            "resolved": self.resolved
        }


class IncidentManager:
    """
    Manages incident tracking and reporting
    From KingSniperV12 with enhancements
    """
    
    def __init__(self, max_incidents: int = 100):
        """
        Initialize incident manager
        
        Args:
            max_incidents: Maximum incidents to keep in memory
        """
        self.incidents: List[Incident] = []
        self.max_incidents = max_incidents
        self._incident_count = 0
    
    def create_incident(
        self,
        session_id: str,
        incident_type: IncidentType,
        severity: IncidentSeverity,
        description: str,
        evidence: Dict[str, Any] = None
    ) -> Incident:
        """
        Create and track a new incident
        
        Args:
            session_id: Session that triggered the incident
            incident_type: Type of incident
            severity: Severity level
            description: Human-readable description
            evidence: Optional evidence dictionary
        
        Returns:
            Created Incident object
        """
        self._incident_count += 1
        incident_id = f"INC-{self._incident_count:05d}-{uuid.uuid4().hex[:6]}"
        
        incident = Incident(
            id=incident_id,
            timestamp=datetime.datetime.now(),
            session_id=session_id,
            type=incident_type,
            severity=severity,
            description=description,
            evidence=evidence or {}
        )
        
        self.incidents.append(incident)
        
        # Trim old incidents
        if len(self.incidents) > self.max_incidents:
            self.incidents = self.incidents[-self.max_incidents:]
        
        # Log based on severity
        log_msg = f"[{incident_id}] {incident_type.value}: {description}"
        if severity == IncidentSeverity.CRITICAL:
            logger.critical(log_msg)
        elif severity == IncidentSeverity.ERROR:
            logger.error(log_msg)
        elif severity == IncidentSeverity.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)
        
        return incident
    
    def resolve_incident(self, incident_id: str) -> bool:
        """Mark an incident as resolved"""
        for incident in self.incidents:
            if incident.id == incident_id:
                incident.resolved = True
                return True
        return False
    
    def get_unresolved(self, session_id: str = None) -> List[Incident]:
        """Get unresolved incidents, optionally filtered by session"""
        incidents = [i for i in self.incidents if not i.resolved]
        if session_id:
            incidents = [i for i in incidents if i.session_id == session_id]
        return incidents
    
    def get_by_type(self, incident_type: IncidentType) -> List[Incident]:
        """Get incidents by type"""
        return [i for i in self.incidents if i.type == incident_type]
    
    def get_recent(self, minutes: int = 5) -> List[Incident]:
        """Get incidents from the last N minutes"""
        cutoff = datetime.datetime.now() - datetime.timedelta(minutes=minutes)
        return [i for i in self.incidents if i.timestamp > cutoff]
    
    def get_summary(self) -> Dict[str, Any]:
        """Get incident summary statistics"""
        summary = {
            "total": len(self.incidents),
            "unresolved": len([i for i in self.incidents if not i.resolved]),
            "by_type": {},
            "by_severity": {}
        }
        
        for incident in self.incidents:
            type_key = incident.type.value
            sev_key = incident.severity.value
            
            summary["by_type"][type_key] = summary["by_type"].get(type_key, 0) + 1
            summary["by_severity"][sev_key] = summary["by_severity"].get(sev_key, 0) + 1
        
        return summary
    
    def export_to_json(self) -> str:
        """Export all incidents to JSON string"""
        return json.dumps([i.to_dict() for i in self.incidents], indent=2)


# ==================== Session Statistics ====================
@dataclass
class SessionStats:
    """Comprehensive session statistics (from KingSniperV12)"""
    # Scanning
    scans: int = 0
    months_scanned: int = 0
    days_found: int = 0
    slots_found: int = 0
    
    # Captcha
    captchas_solved: int = 0
    captchas_failed: int = 0
    
    # Forms
    forms_filled: int = 0
    forms_submitted: int = 0
    
    # Errors
    errors: int = 0
    navigation_errors: int = 0
    
    # Pages
    pages_loaded: int = 0
    
    # Rebirths
    rebirths: int = 0
    session_rebirths: int = 0
    
    # Results
    success: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "scans": self.scans,
            "months_scanned": self.months_scanned,
            "days_found": self.days_found,
            "slots_found": self.slots_found,
            "captchas_solved": self.captchas_solved,
            "captchas_failed": self.captchas_failed,
            "forms_filled": self.forms_filled,
            "forms_submitted": self.forms_submitted,
            "errors": self.errors,
            "navigation_errors": self.navigation_errors,
            "pages_loaded": self.pages_loaded,
            "rebirths": self.rebirths,
            "success": self.success
        }
    
    def get_summary(self) -> str:
        """Get human-readable summary"""
        return (
            f"Scans: {self.scans} | "
            f"Days: {self.days_found} | "
            f"Slots: {self.slots_found} | "
            f"Captchas: {self.captchas_solved}/{self.captchas_failed} | "
            f"Rebirths: {self.rebirths} | "
            f"Errors: {self.errors}"
        )
