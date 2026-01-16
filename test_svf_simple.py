#!/usr/bin/env python3
"""
Simple test for Sky View Factor calculation with artificial data
"""
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import greenshadow as gs
import os

def create_test_dsm():
    """Create a simple test DSM with known features."""
    
    # Create a 100x100 pixel DSM (50x50 meters at 0.5m resolution)
    size = 100
    dsm = np.zeros((size, size), dtype=np.float32)
    
    # Add some simple features
    # 1. A hill in the center
    center = size // 2
    for i in range(size):
        for j in range(size):
            dist_from_center = np.sqrt((i - center)**2 + (j - center)**2)
            if dist_from_center < 15:
                dsm[i, j] = 10 * np.exp(-dist_from_center/10)  # Gaussian hill
    
    # 2. A rectangular building
    dsm[20:30, 20:35] = 15.0
    
    # 3. A wall
    dsm[70:75, 10:80] = 8.0
    
    return dsm

def test_simple_svf():
    """Test SVF with artificial data."""
    
    print("Creating test DSM...")
    dsm = create_test_dsm()
    
    # Define georeferencing (arbitrary coordinates)
    bounds = (0, 0, 50, 50)  # 50x50 meters
    transform = from_bounds(*bounds, dsm.shape[1], dsm.shape[0])
    
    # Create test DSM file
    test_dsm_path = "/tmp/test_dsm.tif"
    
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'count': 1,
        'width': dsm.shape[1],
        'height': dsm.shape[0],
        'transform': transform,
        'crs': 'EPSG:25831',
        'nodata': -9999,
        'compress': 'lzw'
    }
    
    with rasterio.open(test_dsm_path, 'w', **profile) as dst:
        dst.write(dsm, 1)
    
    print(f"Test DSM created: {test_dsm_path}")
    print(f"DSM shape: {dsm.shape}")
    print(f"DSM range: {dsm.min():.2f} to {dsm.max():.2f}")
    
    # Test SVF calculation with faster settings
    print("\\nTesting Sky View Factor calculation...")
    
    svf_path = "/tmp/test_svf.tif"
    
    try:
        result_path = gs.calculate_sky_view_factor(
            dsm_path=test_dsm_path,
            out_path=svf_path,
            search_radius=20.0,      # Small radius for speed
            angular_resolution=15.0,  # Coarse resolution for speed 
            max_height_angle=90.0
        )
        
        print(f"\\nSVF calculation successful!")
        print(f"Result saved to: {result_path}")
        
        # Load and analyze results
        with rasterio.open(svf_path) as src:
            svf_data = src.read(1, masked=True)
            
        print(f"\\nSVF Results:")
        print(f"  Shape: {svf_data.shape}")
        print(f"  Range: {svf_data.min():.3f} to {svf_data.max():.3f}")
        print(f"  Mean: {svf_data.mean():.3f}")
        
        # Check some expected values
        # Open areas should have high SVF (~0.8-1.0)
        # Areas near buildings/hills should have lower SVF
        
        open_area_svf = svf_data[5, 5]  # Corner, should be high
        near_building_svf = svf_data[25, 25]  # Near building, should be lower
        
        print(f"\\nSpot checks:")
        print(f"  Open area SVF (5,5): {open_area_svf:.3f}")
        print(f"  Near building SVF (25,25): {near_building_svf:.3f}")
        
        # Cleanup
        for path in [test_dsm_path, svf_path]:
            if os.path.exists(path):
                os.remove(path)
                
        print("\\nTest completed successfully!")
        
    except Exception as e:
        print(f"Error in SVF calculation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_simple_svf()