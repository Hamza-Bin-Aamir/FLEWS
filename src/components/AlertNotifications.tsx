import { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './AlertNotifications.scss';

interface Alert {
  id: string;
  severity: 'high_risk' | 'medium_risk' | 'low_risk';
  color: 'red' | 'yellow' | 'green';
  location: string;
  region: string;
  coordinates: [number, number];
  message: string;
  issued_at: string;
  issued_minutes_ago: number;
}

interface AlertNotificationsProps {
  alerts: Alert[];
  demoMode?: boolean;
}

const AlertNotifications: React.FC<AlertNotificationsProps> = ({ alerts, demoMode }) => {
  const [visibleAlerts, setVisibleAlerts] = useState<Alert[]>([]);
  const [dismissedIds, setDismissedIds] = useState<Set<string>>(new Set());

  // Reset dismissed alerts when demo mode changes
  useEffect(() => {
    setDismissedIds(new Set());
  }, [demoMode]);

  useEffect(() => {
    // Only show alerts that haven't been dismissed
    setVisibleAlerts(alerts.filter(alert => !dismissedIds.has(alert.id)));
  }, [alerts, dismissedIds]);

  const handleDismiss = (alertId: string) => {
    setDismissedIds(prev => new Set(prev).add(alertId));
  };

  const getSeverityIcon = (severity: string) => {
    switch (severity) {
      case 'high_risk':
        return '🚨';
      case 'medium_risk':
        return '⚠️';
      case 'low_risk':
        return 'ℹ️';
      default:
        return '📢';
    }
  };

  const getSeverityLabel = (severity: string) => {
    switch (severity) {
      case 'high_risk':
        return 'CRITICAL';
      case 'medium_risk':
        return 'WARNING';
      case 'low_risk':
        return 'ADVISORY';
      default:
        return 'INFO';
    }
  };

  const getColorClass = (color: string) => {
    switch (color) {
      case 'red':
        return 'alert-red';
      case 'yellow':
        return 'alert-yellow';
      case 'green':
        return 'alert-green';
      default:
        return 'alert-blue';
    }
  };

  if (visibleAlerts.length === 0) return null;

  return (
    <div className="alert-notifications">
      <AnimatePresence mode="popLayout">
        {visibleAlerts.map((alert) => (
          <motion.div
            key={alert.id}
            initial={{ opacity: 0, x: -50 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -50 }}
            transition={{ duration: 0.3 }}
            className={`alert-notification ${getColorClass(alert.color)}`}
          >
            <div className="alert-icon">
              {getSeverityIcon(alert.severity)}
            </div>
            <div className="alert-content">
              <div className="alert-header">
                <span className="alert-severity">{getSeverityLabel(alert.severity)}</span>
                <span className="alert-time">{alert.issued_minutes_ago}m ago</span>
              </div>
              <div className="alert-location">
                📍 {alert.location}, {alert.region}
              </div>
              <div className="alert-message">
                {alert.message}
              </div>
            </div>
            <button 
              className="alert-dismiss"
              onClick={() => handleDismiss(alert.id)}
              aria-label="Dismiss alert"
            >
              ✕
            </button>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
};

export default AlertNotifications;
