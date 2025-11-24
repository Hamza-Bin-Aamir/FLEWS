"""
Alert management system for FLEWS
Handles live flood alert notifications with stateless, deterministic generation
"""

from typing import List, Dict, Any
from datetime import datetime, timedelta
import random
import hashlib

# Alert locations across Pakistan
ALERT_LOCATIONS = [
    {"name": "Topi, Swabi", "coords": [72.6234, 34.0705], "region": "KPK"},
    {"name": "Nowshera", "coords": [71.9824, 33.9951], "region": "KPK"},
    {"name": "Charsadda", "coords": [71.7417, 34.1483], "region": "KPK"},
    {"name": "Sukkur", "coords": [68.8577, 27.7058], "region": "Sindh"},
    {"name": "Jacobabad", "coords": [68.4375, 28.2769], "region": "Sindh"},
    {"name": "Rahim Yar Khan", "coords": [70.2952, 28.4202], "region": "Punjab"},
    {"name": "Dera Ghazi Khan", "coords": [70.6343, 30.0561], "region": "Punjab"},
    {"name": "Multan", "coords": [71.5249, 30.1575], "region": "Punjab"},
    {"name": "Lahore (Ravi)", "coords": [74.3436, 31.5204], "region": "Punjab"},
    {"name": "Jhelum", "coords": [73.7257, 32.9425], "region": "Punjab"},
    {"name": "Rawalpindi", "coords": [73.0479, 33.5651], "region": "Punjab"},
    {"name": "Karachi Coast", "coords": [67.0099, 24.8607], "region": "Sindh"},
    {"name": "Hyderabad", "coords": [68.3578, 25.3792], "region": "Sindh"},
    {"name": "Muzaffarabad", "coords": [73.4708, 34.3700], "region": "AJK"},
]

ALERT_MESSAGES = {
    "high_risk": [
        "Flash flood warning! Evacuate immediately.",
        "Critical water levels detected. Immediate action required.",
        "Severe flooding in progress. Move to higher ground now.",
        "Emergency: River overflowing. Evacuate low-lying areas.",
        "Danger: Rapid water rise detected. Seek safety immediately.",
    ],
    "medium_risk": [
        "Flood watch in effect. Monitor conditions closely.",
        "Rising water levels detected. Prepare for possible evacuation.",
        "Moderate flood risk. Stay alert and avoid flood-prone areas.",
        "Water levels increasing. Be ready to evacuate if needed.",
        "Flood advisory issued. Exercise caution near waterways.",
    ],
    "low_risk": [
        "Weather advisory: Monitor rainfall conditions.",
        "Minor flooding possible in low-lying areas.",
        "Elevated water levels detected. Stay informed.",
        "Slight flood risk. Avoid unnecessary travel near rivers.",
        "Normal operations with caution advised near water bodies.",
    ]
}

def generate_live_alerts(seed: int = None) -> List[Dict[str, Any]]:
    """
    Generate live alerts using deterministic pseudo-random generation
    Same seed = same alerts (RESTful stateless)
    
    Seed changes every 5 minutes to simulate evolving alert conditions
    """
    # Use seed for deterministic randomness
    if seed is None:
        # Default: change every 5 minutes
        now = datetime.now()
        seed = int(now.strftime("%Y%m%d%H")) * 12 + (now.minute // 5)
    
    # Seed the random generator for deterministic results
    rng = random.Random(seed)
    
    # Deterministic number of active alerts (2-8)
    num_alerts = 2 + (seed % 7)
    alerts = []
    
    # Deterministically select locations
    location_indices = list(range(len(ALERT_LOCATIONS)))
    rng.shuffle(location_indices)
    selected_indices = location_indices[:min(num_alerts, len(ALERT_LOCATIONS))]
    
    for i, idx in enumerate(selected_indices):
        location = ALERT_LOCATIONS[idx]
        
        # Determine severity (deterministic based on seed and index)
        severity_value = (seed + idx * 7) % 10
        if severity_value < 2:  # 20% high risk
            severity = "high_risk"
            color = "red"
        elif severity_value < 5:  # 30% medium risk
            severity = "medium_risk"
            color = "yellow"
        else:  # 50% low risk
            severity = "low_risk"
            color = "green"
        
        # Generate timestamp (deterministic minutes ago)
        issued_minutes_ago = 5 + ((seed + idx * 3) % 25)
        issued_at = datetime.now() - timedelta(minutes=issued_minutes_ago)
        
        # Select message deterministically
        message_idx = (seed + idx * 11) % len(ALERT_MESSAGES[severity])
        
        # Create alert with deterministic ID
        alert_id = hashlib.md5(f"{seed}{idx}{location['name']}".encode()).hexdigest()[:12]
        
        alert = {
            "id": f"alert-{alert_id}",
            "severity": severity,
            "color": color,
            "location": location["name"],
            "region": location["region"],
            "coordinates": location["coords"],
            "message": ALERT_MESSAGES[severity][message_idx],
            "issued_at": issued_at.isoformat(),
            "issued_minutes_ago": issued_minutes_ago,
        }
        alerts.append(alert)
    
    # Sort by severity (high first, then medium, then low)
    severity_order = {"high_risk": 0, "medium_risk": 1, "low_risk": 2}
    alerts.sort(key=lambda x: severity_order[x["severity"]])
    
    return alerts

def get_alert_summary(seed: int = None) -> Dict[str, Any]:
    """
    Returns live alerts with full information
    
    RESTful: Same seed always returns same alerts (deterministic)
    Seed changes every 5 minutes to simulate evolving conditions
    
    Args:
        seed: Optional seed for deterministic generation
    """
    alerts = generate_live_alerts(seed)
    
    # Count by severity
    high_count = sum(1 for a in alerts if a["severity"] == "high_risk")
    medium_count = sum(1 for a in alerts if a["severity"] == "medium_risk")
    low_count = sum(1 for a in alerts if a["severity"] == "low_risk")
    
    return {
        "alerts": alerts,
        "summary": {
            "high_risk": high_count,
            "medium_risk": medium_count,
            "low_risk": low_count,
            "total": len(alerts)
        },
        "last_updated": datetime.now().isoformat()
    }


def create_alert(location: str, severity: str, message: str) -> Dict:
    """
    Create a new flood alert
    
    Args:
        location: Location name or description
        severity: Alert severity level (high, moderate, low)
        message: Alert message content
    
    Returns:
        Created alert object
    """
    # Dummy implementation
    alert = {
        "id": f"alert_{datetime.now().timestamp()}",
        "location": location,
        "severity": severity,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "status": "active"
    }
    return alert


def dismiss_alert(alert_id: str) -> bool:
    """
    Dismiss or deactivate an alert
    
    Args:
        alert_id: ID of the alert to dismiss
    
    Returns:
        True if successful, False otherwise
    """
    # Dummy implementation
    return True
