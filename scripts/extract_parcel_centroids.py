#!/usr/bin/env python3
"""
Extract parcel centroids from King County GIS GeoJSON.

This script parses the large GeoJSON file line-by-line to extract
MAJOR, MINOR, and calculate centroid lat/long for each parcel.
"""

import json
import re
import csv
from pathlib import Path
from typing import Generator


def calculate_centroid(coordinates: list) -> tuple[float, float]:
    """
    Calculate centroid of a MultiPolygon.
    
    For simplicity, uses the centroid of the first polygon's exterior ring.
    Returns (longitude, latitude).
    """
    # MultiPolygon structure: [[[ring1], [ring2], ...], ...]
    # We want the first polygon's first ring (exterior)
    try:
        if not coordinates:
            return (None, None)
        
        # Get first polygon's exterior ring
        first_polygon = coordinates[0]
        if not first_polygon:
            return (None, None)
            
        exterior_ring = first_polygon[0]
        if not exterior_ring:
            return (None, None)
        
        # Calculate centroid as average of all points
        lons = [pt[0] for pt in exterior_ring]
        lats = [pt[1] for pt in exterior_ring]
        
        centroid_lon = sum(lons) / len(lons)
        centroid_lat = sum(lats) / len(lats)
        
        return (centroid_lon, centroid_lat)
    except (IndexError, TypeError, ZeroDivisionError):
        return (None, None)


def stream_geojson_features(filepath: Path) -> Generator[dict, None, None]:
    """
    Stream features from a large GeoJSON file.
    
    Uses regex to find feature boundaries rather than loading entire file.
    """
    print(f"Streaming features from {filepath}...")
    
    # Read in chunks and parse features
    feature_pattern = re.compile(
        r'\{ "type": "Feature", "properties": \{([^}]+)\}, "geometry": \{([^}]+)\} \}'
    )
    
    count = 0
    with open(filepath, 'r') as f:
        # Skip to features array
        buffer = ""
        chunk_size = 10 * 1024 * 1024  # 10MB chunks
        
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            
            buffer += chunk
            
            # Find complete features in buffer
            # Look for pattern: { "type": "Feature" ... } },
            start = 0
            while True:
                # Find start of feature
                feat_start = buffer.find('{ "type": "Feature"', start)
                if feat_start == -1:
                    break
                
                # Find end of feature (next feature start or end of features)
                next_feat = buffer.find('{ "type": "Feature"', feat_start + 1)
                
                if next_feat == -1:
                    # Might be incomplete, keep in buffer
                    buffer = buffer[feat_start:]
                    break
                
                # Extract feature JSON
                feature_str = buffer[feat_start:next_feat].rstrip().rstrip(',')
                
                try:
                    feature = json.loads(feature_str)
                    count += 1
                    if count % 100000 == 0:
                        print(f"  Processed {count:,} features...")
                    yield feature
                except json.JSONDecodeError:
                    pass  # Skip malformed features
                
                start = next_feat
            
            # Keep unparsed portion for next iteration
            if start > 0:
                buffer = buffer[start:]
    
    print(f"  Total features processed: {count:,}")


def extract_centroids_simple(filepath: Path, output_path: Path):
    """
    Simple line-by-line extraction of parcel centroids.
    """
    print(f"Extracting centroids from {filepath}...")
    print(f"File size: {filepath.stat().st_size / 1024 / 1024:.1f} MB")
    
    centroids = []
    count = 0
    
    with open(filepath, 'r') as f:
        for line in f:
            # Look for feature lines
            if '"MAJOR"' in line and '"geometry"' in line:
                try:
                    # Try to parse as JSON (strip trailing comma if present)
                    line = line.strip().rstrip(',')
                    feature = json.loads(line)
                    
                    props = feature.get('properties', {})
                    geom = feature.get('geometry', {})
                    
                    major = props.get('MAJOR', '')
                    minor = props.get('MINOR', '')
                    coords = geom.get('coordinates', [])
                    
                    lon, lat = calculate_centroid(coords)
                    
                    if major and minor and lon and lat:
                        centroids.append({
                            'Major': str(major),
                            'Minor': str(minor).zfill(4),
                            'lat': lat,
                            'long': lon
                        })
                        count += 1
                        
                        if count % 100000 == 0:
                            print(f"  Extracted {count:,} centroids...")
                            
                except json.JSONDecodeError:
                    continue
    
    print(f"\nTotal centroids extracted: {count:,}")
    
    # Write to CSV
    print(f"Writing to {output_path}...")
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['Major', 'Minor', 'lat', 'long'])
        writer.writeheader()
        writer.writerows(centroids)
    
    print(f"Done! File size: {output_path.stat().st_size / 1024:.1f} KB")
    
    return centroids


def main():
    geojson_path = Path("Reference_Docs/King_County_Parcels___parcel_area.geojson")
    output_path = Path("data/parcel_centroids.csv")
    
    if not geojson_path.exists():
        print(f"ERROR: GeoJSON file not found: {geojson_path}")
        return
    
    centroids = extract_centroids_simple(geojson_path, output_path)
    
    # Quick stats
    if centroids:
        lats = [c['lat'] for c in centroids]
        lons = [c['long'] for c in centroids]
        print(f"\nCentroid Statistics:")
        print(f"  Latitude range:  {min(lats):.4f} to {max(lats):.4f}")
        print(f"  Longitude range: {min(lons):.4f} to {max(lons):.4f}")


if __name__ == "__main__":
    main()
