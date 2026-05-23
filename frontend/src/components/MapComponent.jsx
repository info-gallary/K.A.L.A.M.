 
import { useEffect, useRef, useState } from 'react';
import mapboxgl from 'mapbox-gl';
import 'mapbox-gl/dist/mapbox-gl.css';

mapboxgl.accessToken = import.meta.env.VITE_MAPBOX_ACCESS_TOKEN;

export default function MapComponent() {
  const mapContainer = useRef(null);
  const map = useRef(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!mapContainer.current) return;

    try {
      mapboxgl.config.REQUIRE_ACCESS_TOKEN = true;

      map.current = new mapboxgl.Map({
        container: mapContainer.current,
        style:'mapbox://styles/mapbox/satellite-streets-v12',
        center: [80.7, 21], // Center over India
        zoom: 3,
        projection: 'mercator',
        minZoom: 2,
        maxZoom: 6.5,
        maxBounds: [
          [60, 0],     // Southwest (longitude, latitude)
          [100, 38]    // Northeast (longitude, latitude)
        ]
        // Removed maxBounds for full pan freedom
      });
 
      map.current.addControl(new mapboxgl.NavigationControl(), 'top-right');
      
      map.current.on('zoom', () => {
        const currentZoom = map.current.getZoom().toFixed(2);
        console.log('📦 Current Zoom Level:', currentZoom);
      });



      map.current.on('error', (e) => {
        console.error('Mapbox Error:', e.error);
        if (e?.error?.message?.includes('<!DOCTYPE')) {
          setError(
            'Custom style not loading – received HTML instead of JSON. Check token or style access.'
          );
        }
      });

      map.current.on('load', () => {
        // Example raster overlay
        // map.current.addSource('cloud-raster', {
        //   type: 'raster',
        //   url: 'mapbox://krishv0610.cloud_mask_20250806_2109',
        //   tileSize: 256
        // });

        // map.current.addLayer({
        //   id: 'cloud-overlay',
        //   type: 'raster',
        //   source: 'cloud-raster',
        //   paint: {
        //     'raster-opacity': 1
        //   }
        // });

        map.current.jumpTo({
          center: [80.7, 21],
        });

        addStyleSwitcher();
      });
    } catch (err) {
      setError(`Map initialization failed: ${err.message}`);
      console.error('Map error:', err);
    }

    return () => {
      if (map.current) map.current.remove();
    };
  }, []);

  const addStyleSwitcher = () => {
    const controlDiv = document.createElement('div');
    controlDiv.className = 'mapboxgl-ctrl mapboxgl-ctrl-group';
    controlDiv.style.background = '#fff';

    const styles = {
      Hybrid: 'mapbox://styles/mapbox/satellite-streets-v12',
      Custom:'mapbox://styles/krishv0610/cmdo9yevo002v01sbetze2uc0'
    };

    Object.entries(styles).forEach(([label, styleUrl]) => {
      const btn = document.createElement('button');
      btn.textContent = label;
      btn.title = `Switch to ${label} view`;
      btn.onclick = () => {
        const currentCenter = map.current.getCenter();
        const currentZoom = map.current.getZoom();
        const currentPitch = map.current.getPitch();
        const currentBearing = map.current.getBearing();

        map.current.setStyle(styleUrl);
        map.current.once('styledata', () => {
          map.current.jumpTo({
            center: currentCenter,
            zoom: currentZoom,
            pitch: currentPitch,
            bearing: currentBearing
          });
        });
      };
      controlDiv.appendChild(btn);
    });

    map.current.addControl(
      {
        onAdd: () => controlDiv,
        onRemove: () => controlDiv.remove()
      },
      'top-left'
    );
  };

  if (error) {
    return (
      <div style={{ padding: '20px', color: 'red' }}>
        <h3>Error Loading Map</h3>
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div className="flex justify-end items-start w-[100%] h-[100%] bg-gray-100 p-2">
      <div className="w-[100%] h-[100%]  shadow rounded-2xl overflow-hidden">
        <div ref={mapContainer} className="w-full h-full" />
      </div>
   
    </div>
  );
}
