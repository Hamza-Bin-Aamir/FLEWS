import { useState, useRef, useEffect } from 'react';
import './MapSearch.scss';

interface SearchResult {
  id: string;
  place_name: string;
  center: [number, number];
}

interface MapSearchProps {
  onLocationSelect: (coordinates: [number, number], placeName: string) => void;
  mapboxToken: string;
  currentCenter?: [number, number];
}

const MapSearch: React.FC<MapSearchProps> = ({ onLocationSelect, mapboxToken, currentCenter }) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [showResults, setShowResults] = useState(false);
  const searchTimeoutRef = useRef<number | undefined>(undefined);

  useEffect(() => {
    // Clear timeout on unmount
    return () => {
      if (searchTimeoutRef.current) {
        clearTimeout(searchTimeoutRef.current);
      }
    };
  }, []);

  const handleSearch = async (query: string) => {
    setSearchQuery(query);

    // Clear previous timeout
    if (searchTimeoutRef.current) {
      clearTimeout(searchTimeoutRef.current);
    }

    if (query.trim().length < 3) {
      setSearchResults([]);
      setShowResults(false);
      return;
    }

    // Debounce search
    searchTimeoutRef.current = setTimeout(async () => {
      setIsLoading(true);
      try {
        // Build proximity parameter if current center is available
        const proximityParam = currentCenter 
          ? `&proximity=${currentCenter[0]},${currentCenter[1]}`
          : '';

        const response = await fetch(
          `https://api.mapbox.com/geocoding/v5/mapbox.places/${encodeURIComponent(
            query
          )}.json?access_token=${mapboxToken}&limit=5${proximityParam}`
        );

        if (!response.ok) {
          throw new Error('Failed to fetch search results');
        }

        const data = await response.json();
        setSearchResults(data.features || []);
        setShowResults(true);
      } catch (error) {
        console.error('Search error:', error);
        setSearchResults([]);
      } finally {
        setIsLoading(false);
      }
    }, 500);
  };

  const handleSelectLocation = (result: SearchResult) => {
    onLocationSelect(result.center, result.place_name);
    setSearchQuery(result.place_name);
    setShowResults(false);
    setSearchResults([]);
  };

  const handleClearSearch = () => {
    setSearchQuery('');
    setSearchResults([]);
    setShowResults(false);
  };

  return (
    <div className="map-search">
      <div className="search-input-wrapper">
        <span className="search-icon">🔍</span>
        <input
          type="text"
          className="search-input"
          placeholder="Search for locations..."
          value={searchQuery}
          onChange={(e) => handleSearch(e.target.value)}
          onFocus={() => searchResults.length > 0 && setShowResults(true)}
        />
        {searchQuery && (
          <button className="clear-button" onClick={handleClearSearch}>
            ✕
          </button>
        )}
        {isLoading && (
          <div className="search-spinner"></div>
        )}
      </div>

      {showResults && searchResults.length > 0 && (
        <div className="search-results">
          {searchResults.map((result) => (
            <div
              key={result.id}
              className="search-result-item"
              onClick={() => handleSelectLocation(result)}
            >
              <span className="result-icon">📍</span>
              <span className="result-text">{result.place_name}</span>
            </div>
          ))}
        </div>
      )}

      {showResults && !isLoading && searchResults.length === 0 && searchQuery.length >= 3 && (
        <div className="search-results">
          <div className="no-results">
            <span>No locations found</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default MapSearch;
