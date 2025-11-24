import { motion } from 'framer-motion';
import MapBackground from './MapBackground';
import './LandingPage.scss';

const LandingPage = () => {
  return (
    <div className="dashboard">
      {/* Header */}
      <motion.header
        className="dashboard-header"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="header-content">
          <div className="logo-section">
            <h1 className="logo">🌊 FLEWS</h1>
            <span className="subtitle">Flood Early Warning System</span>
          </div>
        </div>
      </motion.header>

      {/* Map Container - The central interactive element */}
      <div className="map-view">
        <MapBackground />
      </div>
    </div>
  );
};

export default LandingPage;
