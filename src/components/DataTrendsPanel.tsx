import React, { useEffect, useState, useRef } from 'react';
import * as d3 from 'd3';
import { useCombinedTrends } from '../hooks/useTrendsData';
import './DataTrendsPanel.scss';

interface DataTrendsPanelProps {
  lat: number;
  lon: number;
  onClose?: () => void;
}

type TabType = 'rainfall' | 'river' | 'combined';

const DataTrendsPanel: React.FC<DataTrendsPanelProps> = ({ lat, lon, onClose }) => {
  const [activeTab, setActiveTab] = useState<TabType>('rainfall');
  const [days, setDays] = useState(7);
  const { data, loading, error, fetchCombinedTrends } = useCombinedTrends();
  
  const rainfallChartRef = useRef<SVGSVGElement>(null);
  const riverChartRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (lat && lon) {
      fetchCombinedTrends(lat, lon, days);
    }
  }, [lat, lon, days, fetchCombinedTrends]);

  // Draw rainfall chart using D3
  useEffect(() => {
    if (!data?.rainfall?.historical || !rainfallChartRef.current) return;
    
    const svg = d3.select(rainfallChartRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = 500 - margin.left - margin.right;
    const height = 250 - margin.top - margin.bottom;

    const g = svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Combine historical and forecast data
    const historicalData = data.rainfall.historical.map(d => ({
      date: new Date(d.timestamp),
      value: d.rainfall_1h,
      type: 'historical' as const
    }));

    const forecastData = data.rainfall.forecast.map(d => ({
      date: new Date(d.timestamp),
      value: d.rainfall_3h / 3, // Convert 3h to 1h equivalent
      type: 'forecast' as const
    }));

    const allData = [...historicalData, ...forecastData];

    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(allData, d => d.date) as [Date, Date])
      .range([0, width]);

    const yScale = d3.scaleLinear()
      .domain([0, d3.max(allData, d => d.value) || 10])
      .nice()
      .range([height, 0]);

    // Grid lines
    g.append('g')
      .attr('class', 'grid')
      .attr('opacity', 0.1)
      .call(d3.axisLeft(yScale)
        .tickSize(-width)
        .tickFormat(() => '')
      );

    // Area for historical data
    const historicalArea = d3.area<typeof historicalData[0]>()
      .x(d => xScale(d.date))
      .y0(height)
      .y1(d => yScale(d.value))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(historicalData)
      .attr('fill', 'rgba(59, 130, 246, 0.3)')
      .attr('d', historicalArea);

    // Line for historical data
    const line = d3.line<typeof historicalData[0]>()
      .x(d => xScale(d.date))
      .y(d => yScale(d.value))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(historicalData)
      .attr('fill', 'none')
      .attr('stroke', '#3b82f6')
      .attr('stroke-width', 2)
      .attr('d', line);

    // Dashed line for forecast
    g.append('path')
      .datum(forecastData)
      .attr('fill', 'none')
      .attr('stroke', '#f59e0b')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5')
      .attr('d', line);

    // Forecast area (with lower opacity)
    g.append('path')
      .datum(forecastData)
      .attr('fill', 'rgba(245, 158, 11, 0.2)')
      .attr('d', historicalArea);

    // Now/forecast divider line
    if (historicalData.length > 0 && forecastData.length > 0) {
      const nowX = xScale(historicalData[historicalData.length - 1].date);
      g.append('line')
        .attr('x1', nowX)
        .attr('x2', nowX)
        .attr('y1', 0)
        .attr('y2', height)
        .attr('stroke', '#fff')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3')
        .attr('opacity', 0.5);

      g.append('text')
        .attr('x', nowX)
        .attr('y', -5)
        .attr('text-anchor', 'middle')
        .attr('fill', '#fff')
        .attr('font-size', '10px')
        .text('Now');
    }

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .attr('class', 'axis')
      .call(d3.axisBottom(xScale)
        .ticks(6)
        .tickFormat(d => d3.timeFormat('%b %d')(d as Date))
      )
      .selectAll('text')
      .attr('fill', '#94a3b8');

    g.append('g')
      .attr('class', 'axis')
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text')
      .attr('fill', '#94a3b8');

    // Y-axis label
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -40)
      .attr('x', -height / 2)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8')
      .attr('font-size', '12px')
      .text('Rainfall (mm/h)');

    // Style axis lines
    svg.selectAll('.axis path, .axis line')
      .attr('stroke', '#475569');

  }, [data?.rainfall]);

  // Draw river level chart using D3
  useEffect(() => {
    if (!data?.river_level?.historical || !riverChartRef.current) return;
    
    const svg = d3.select(riverChartRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 20, right: 30, bottom: 40, left: 50 };
    const width = 500 - margin.left - margin.right;
    const height = 250 - margin.top - margin.bottom;

    const g = svg
      .attr('width', width + margin.left + margin.right)
      .attr('height', height + margin.top + margin.bottom)
      .append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    const historicalData = data.river_level.historical.map(d => ({
      date: new Date(d.timestamp),
      value: d.level,
      status: d.status
    }));

    const forecastData = data.river_level.forecast.map(d => ({
      date: new Date(d.timestamp),
      value: d.level,
      min: d.level_min,
      max: d.level_max,
      status: d.status
    }));

    const allData = [...historicalData, ...forecastData];
    const thresholds = data.river_level.thresholds;

    // Scales
    const xScale = d3.scaleTime()
      .domain(d3.extent(allData, d => d.date) as [Date, Date])
      .range([0, width]);

    const maxLevel = Math.max(
      d3.max(allData, d => d.value) || 10,
      thresholds.extreme + 1
    );

    const yScale = d3.scaleLinear()
      .domain([0, maxLevel])
      .nice()
      .range([height, 0]);

    // Threshold zones
    const thresholdZones = [
      { y: yScale(thresholds.extreme), height: yScale(0) - yScale(thresholds.extreme), color: 'rgba(239, 68, 68, 0.15)', label: 'Extreme' },
      { y: yScale(thresholds.danger), height: yScale(thresholds.extreme) - yScale(thresholds.danger), color: 'rgba(249, 115, 22, 0.15)', label: 'Danger' },
      { y: yScale(thresholds.warning), height: yScale(thresholds.danger) - yScale(thresholds.warning), color: 'rgba(245, 158, 11, 0.15)', label: 'Warning' },
      { y: yScale(thresholds.normal), height: yScale(thresholds.warning) - yScale(thresholds.normal), color: 'rgba(34, 197, 94, 0.1)', label: 'Normal' },
    ];

    thresholdZones.forEach(zone => {
      g.append('rect')
        .attr('x', 0)
        .attr('y', zone.y)
        .attr('width', width)
        .attr('height', zone.height)
        .attr('fill', zone.color);
    });

    // Threshold lines
    const thresholdLines = [
      { value: thresholds.warning, color: '#f59e0b', label: 'Warning' },
      { value: thresholds.danger, color: '#f97316', label: 'Danger' },
      { value: thresholds.extreme, color: '#ef4444', label: 'Extreme' },
    ];

    thresholdLines.forEach(th => {
      g.append('line')
        .attr('x1', 0)
        .attr('x2', width)
        .attr('y1', yScale(th.value))
        .attr('y2', yScale(th.value))
        .attr('stroke', th.color)
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '4,4')
        .attr('opacity', 0.7);

      g.append('text')
        .attr('x', width - 5)
        .attr('y', yScale(th.value) - 3)
        .attr('text-anchor', 'end')
        .attr('fill', th.color)
        .attr('font-size', '9px')
        .text(th.label);
    });

    // Forecast confidence band
    if (forecastData.length > 0) {
      const area = d3.area<typeof forecastData[0]>()
        .x(d => xScale(d.date))
        .y0(d => yScale(d.min))
        .y1(d => yScale(d.max))
        .curve(d3.curveMonotoneX);

      g.append('path')
        .datum(forecastData)
        .attr('fill', 'rgba(139, 92, 246, 0.2)')
        .attr('d', area);
    }

    // Historical line
    const line = d3.line<{ date: Date; value: number }>()
      .x(d => xScale(d.date))
      .y(d => yScale(d.value))
      .curve(d3.curveMonotoneX);

    g.append('path')
      .datum(historicalData)
      .attr('fill', 'none')
      .attr('stroke', '#22c55e')
      .attr('stroke-width', 2.5)
      .attr('d', line);

    // Forecast line
    g.append('path')
      .datum(forecastData)
      .attr('fill', 'none')
      .attr('stroke', '#8b5cf6')
      .attr('stroke-width', 2)
      .attr('stroke-dasharray', '5,5')
      .attr('d', line);

    // Now divider
    if (historicalData.length > 0 && forecastData.length > 0) {
      const nowX = xScale(historicalData[historicalData.length - 1].date);
      g.append('line')
        .attr('x1', nowX)
        .attr('x2', nowX)
        .attr('y1', 0)
        .attr('y2', height)
        .attr('stroke', '#fff')
        .attr('stroke-width', 1)
        .attr('stroke-dasharray', '3,3')
        .attr('opacity', 0.5);

      g.append('text')
        .attr('x', nowX)
        .attr('y', -5)
        .attr('text-anchor', 'middle')
        .attr('fill', '#fff')
        .attr('font-size', '10px')
        .text('Now');
    }

    // Axes
    g.append('g')
      .attr('transform', `translate(0,${height})`)
      .attr('class', 'axis')
      .call(d3.axisBottom(xScale)
        .ticks(6)
        .tickFormat(d => d3.timeFormat('%b %d')(d as Date))
      )
      .selectAll('text')
      .attr('fill', '#94a3b8');

    g.append('g')
      .attr('class', 'axis')
      .call(d3.axisLeft(yScale).ticks(5))
      .selectAll('text')
      .attr('fill', '#94a3b8');

    // Y-axis label
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -40)
      .attr('x', -height / 2)
      .attr('text-anchor', 'middle')
      .attr('fill', '#94a3b8')
      .attr('font-size', '12px')
      .text('Water Level (m)');

    svg.selectAll('.axis path, .axis line')
      .attr('stroke', '#475569');

  }, [data?.river_level]);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'Normal': return '#22c55e';
      case 'Warning': return '#f59e0b';
      case 'Danger': return '#f97316';
      case 'Extreme': return '#ef4444';
      case 'Safe': return '#22c55e';
      case 'At Risk': return '#f59e0b';
      case 'Flooded': return '#ef4444';
      default: return '#94a3b8';
    }
  };

  const getTrendIcon = (trend: string) => {
    switch (trend) {
      case 'rising': return '📈';
      case 'falling': return '📉';
      default: return '➡️';
    }
  };

  if (loading) {
    return (
      <div className="data-trends-panel">
        <div className="panel-header">
          <h3>📊 Data Trends & Analytics</h3>
          {onClose && <button className="close-btn" onClick={onClose}>×</button>}
        </div>
        <div className="loading">
          <div className="spinner"></div>
          <p>Loading trend data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="data-trends-panel">
        <div className="panel-header">
          <h3>📊 Data Trends & Analytics</h3>
          {onClose && <button className="close-btn" onClick={onClose}>×</button>}
        </div>
        <div className="error">
          <p>⚠️ {error}</p>
          <button onClick={() => fetchCombinedTrends(lat, lon, days)}>Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="data-trends-panel">
      <div className="panel-header">
        <h3>📊 Data Trends & Analytics</h3>
        {onClose && <button className="close-btn" onClick={onClose}>×</button>}
      </div>

      {/* Time Range Selector */}
      <div className="time-selector">
        <label>Time Range:</label>
        <div className="time-buttons">
          <button 
            className={days === 3 ? 'active' : ''} 
            onClick={() => setDays(3)}
          >
            3D
          </button>
          <button 
            className={days === 7 ? 'active' : ''} 
            onClick={() => setDays(7)}
          >
            7D
          </button>
          <button 
            className={days === 14 ? 'active' : ''} 
            onClick={() => setDays(14)}
          >
            14D
          </button>
          <button 
            className={days === 30 ? 'active' : ''} 
            onClick={() => setDays(30)}
          >
            30D
          </button>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="tab-navigation">
        <button 
          className={activeTab === 'rainfall' ? 'active' : ''}
          onClick={() => setActiveTab('rainfall')}
        >
          🌧️ Rainfall
        </button>
        <button 
          className={activeTab === 'river' ? 'active' : ''}
          onClick={() => setActiveTab('river')}
        >
          🌊 River Level
        </button>
        <button 
          className={activeTab === 'combined' ? 'active' : ''}
          onClick={() => setActiveTab('combined')}
        >
          📈 Combined
        </button>
      </div>

      {/* Rainfall Tab */}
      {activeTab === 'rainfall' && data?.rainfall && (
        <div className="tab-content">
          <div className="chart-container">
            <h4>Rainfall History & Forecast</h4>
            <svg ref={rainfallChartRef}></svg>
            <div className="chart-legend">
              <span className="legend-item">
                <span className="legend-color" style={{ backgroundColor: '#3b82f6' }}></span>
                Historical
              </span>
              <span className="legend-item">
                <span className="legend-color" style={{ backgroundColor: '#f59e0b' }}></span>
                Forecast
              </span>
            </div>
          </div>

          {data.rainfall.summary && (
            <div className="summary-cards">
              <div className="summary-card">
                <span className="card-icon">💧</span>
                <div className="card-content">
                  <span className="card-value">{data.rainfall.summary.total_rainfall} mm</span>
                  <span className="card-label">Total Rainfall</span>
                </div>
              </div>
              <div className="summary-card">
                <span className="card-icon">📊</span>
                <div className="card-content">
                  <span className="card-value">{data.rainfall.summary.avg_rainfall} mm/h</span>
                  <span className="card-label">Average</span>
                </div>
              </div>
              <div className="summary-card">
                <span className="card-icon">⬆️</span>
                <div className="card-content">
                  <span className="card-value">{data.rainfall.summary.max_rainfall} mm/h</span>
                  <span className="card-label">Maximum</span>
                </div>
              </div>
              <div className="summary-card warning">
                <span className="card-icon">⚠️</span>
                <div className="card-content">
                  <span className="card-value">{data.rainfall.summary.flood_events}</span>
                  <span className="card-label">Flood Events</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* River Level Tab */}
      {activeTab === 'river' && data?.river_level && (
        <div className="tab-content">
          <div className="river-header">
            <h4>🏞️ {data.river_level.river_name}</h4>
            {data.river_level.summary && (
              <span 
                className="current-status"
                style={{ backgroundColor: getStatusColor(data.river_level.summary.current_status) }}
              >
                {data.river_level.summary.current_status}
              </span>
            )}
          </div>

          <div className="chart-container">
            <h4>Water Level History & Forecast</h4>
            <svg ref={riverChartRef}></svg>
            <div className="chart-legend">
              <span className="legend-item">
                <span className="legend-color" style={{ backgroundColor: '#22c55e' }}></span>
                Historical
              </span>
              <span className="legend-item">
                <span className="legend-color" style={{ backgroundColor: '#8b5cf6' }}></span>
                Forecast
              </span>
              <span className="legend-item">
                <span className="legend-color" style={{ backgroundColor: 'rgba(139, 92, 246, 0.3)' }}></span>
                Confidence Range
              </span>
            </div>
          </div>

          {data.river_level.summary && (
            <div className="summary-cards">
              <div className="summary-card">
                <span className="card-icon">🌊</span>
                <div className="card-content">
                  <span className="card-value">{data.river_level.summary.current_level} m</span>
                  <span className="card-label">Current Level</span>
                </div>
              </div>
              <div className="summary-card">
                <span className="card-icon">{getTrendIcon(data.river_level.summary.trend)}</span>
                <div className="card-content">
                  <span className="card-value" style={{ textTransform: 'capitalize' }}>
                    {data.river_level.summary.trend}
                  </span>
                  <span className="card-label">Trend</span>
                </div>
              </div>
              <div className="summary-card">
                <span className="card-icon">⬆️</span>
                <div className="card-content">
                  <span className="card-value">{data.river_level.summary.max_level} m</span>
                  <span className="card-label">Max Level</span>
                </div>
              </div>
              <div className="summary-card">
                <span className="card-icon">📊</span>
                <div className="card-content">
                  <span className="card-value">{data.river_level.summary.avg_level} m</span>
                  <span className="card-label">Average</span>
                </div>
              </div>
            </div>
          )}

          <div className="threshold-legend">
            <h5>Risk Thresholds</h5>
            <div className="thresholds">
              <span className="threshold-item" style={{ color: '#22c55e' }}>
                Normal: {'<'} {data.river_level.thresholds.warning}m
              </span>
              <span className="threshold-item" style={{ color: '#f59e0b' }}>
                Warning: {data.river_level.thresholds.warning}m
              </span>
              <span className="threshold-item" style={{ color: '#f97316' }}>
                Danger: {data.river_level.thresholds.danger}m
              </span>
              <span className="threshold-item" style={{ color: '#ef4444' }}>
                Extreme: {data.river_level.thresholds.extreme}m
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Combined Tab */}
      {activeTab === 'combined' && data && (
        <div className="tab-content">
          <div className="correlation-section">
            <h4>📊 Rainfall & River Level Correlation</h4>
            <div className="correlation-indicator">
              <div className="correlation-value">
                <span className="value">{(data.correlation.coefficient * 100).toFixed(1)}%</span>
                <span className="label">Correlation</span>
              </div>
              <p className="correlation-description">{data.correlation.description}</p>
            </div>
          </div>

          <div className="mini-charts">
            <div className="mini-chart">
              <h5>🌧️ Rainfall Trend</h5>
              <svg ref={rainfallChartRef}></svg>
            </div>
            <div className="mini-chart">
              <h5>🌊 River Level Trend</h5>
              <svg ref={riverChartRef}></svg>
            </div>
          </div>

          <div className="combined-summary">
            <h5>Key Insights</h5>
            <ul>
              {data.rainfall?.summary && (
                <li>
                  <strong>Rainfall:</strong> {data.rainfall.summary.total_rainfall}mm total, 
                  {data.rainfall.summary.flood_events} flood events in the period
                </li>
              )}
              {data.river_level?.summary && (
                <li>
                  <strong>River Level:</strong> Currently {data.river_level.summary.current_level}m 
                  ({data.river_level.summary.current_status}), trending {data.river_level.summary.trend}
                </li>
              )}
              <li>
                <strong>Risk Assessment:</strong> {
                  data.correlation.coefficient > 0.5 
                    ? 'High rainfall is strongly linked to rising river levels - monitor closely during rain'
                    : 'Multiple factors affect river levels - consider upstream conditions'
                }
              </li>
            </ul>
          </div>
        </div>
      )}

      <div className="panel-footer">
        <span className="last-updated">
          Last updated: {data?.timestamp ? new Date(data.timestamp).toLocaleString() : 'N/A'}
        </span>
        <button 
          className="refresh-btn"
          onClick={() => fetchCombinedTrends(lat, lon, days)}
        >
          🔄 Refresh
        </button>
      </div>
    </div>
  );
};

export default DataTrendsPanel;