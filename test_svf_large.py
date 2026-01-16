#!/usr/bin/env python3
"""
Test Sky View Factor calculation with larger dataset to test interpolation
"""
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import greenshadow as gs
import os

def create_large_test_dsm():
    """Create a larger test DSM to trigger downsampling."""
    
    # Create a 200x200 pixel DSM (will trigger downsampling)
    size = 200
    dsm = np.zeros((size, size), dtype=np.float32)
    
    # Add some complex features
    center = size // 2
    
    # 1. Rolling hills
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center
            # Multiple hills with different heights
            hill1 = 8 * np.exp(-((x-30)**2 + (y-20)**2)/300)
            hill2 = 12 * np.exp(-((x+40)**2 + (y-30)**2)/400) 
            hill3 = 6 * np.exp(-((x-10)**2 + (y+40)**2)/200)
            dsm[i, j] = hill1 + hill2 + hill3
    
    # 2. Some buildings
    dsm[50:70, 50:80] = 20.0    # Large building
    dsm[120:135, 60:90] = 15.0  # Medium building
    dsm[30:40, 150:170] = 25.0  # Tall building
    
    # 3. A ridge/wall
    dsm[100:110, 20:180] = 18.0
    
    return dsm

def test_large_svf():
    """Test SVF with larger data to verify interpolation."""
    
    print("Creating large test DSM...")
    dsm = create_large_test_dsm()
    
    # Define georeferencing
    bounds = (0, 0, 100, 100)  # 100x100 meters  
    transform = from_bounds(*bounds, dsm.shape[1], dsm.shape[0])
    
    # Create test DSM file
    test_dsm_path = "/tmp/test_dsm_large.tif"
    
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'count': 1,
        'width': dsm.shape[1],
        'height': dsm.shape[0],
        'transform': transform,
        'crs': 'EPSG:25831',
        'nodata': -9999
    }
    
    with rasterio.open(test_dsm_path, 'w', **profile) as dst:
        dst.write(dsm, 1)
    
    print(f"Large test DSM created: {test_dsm_path}")
    print(f"DSM shape: {dsm.shape}")
    print(f"DSM range: {dsm.min():.2f} to {dsm.max():.2f}")
    
    # Test SVF calculation
    print("\\nTesting SVF calculation with interpolation...")
    
    svf_path = "/tmp/test_svf_large.tif"
    
    try:
        result_path = gs.calculate_sky_view_factor(
            dsm_path=test_dsm_path,
            out_path=svf_path,
            search_radius=30.0,      # Larger radius
            angular_resolution=10.0,  # Reasonable resolution
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
        
        # Check interpolation quality by sampling different areas
        print(f"\\nSpatial variation analysis:")
        
        # Sample different regions
        open_area = svf_data[10, 10]  # Corner area
        near_building = svf_data[60, 65]  # Near large building
        on_ridge = svf_data[105, 100]  # On the ridge
        valley = svf_data[100-50, 100]  # In a valley between hills
        
        print(f"  Open area SVF: {open_area:.3f}")
        print(f"  Near building SVF: {near_building:.3f}")
        print(f"  On ridge SVF: {on_ridge:.3f}")  
        print(f"  Valley SVF: {valley:.3f}")
        
        # Check for interpolation artifacts (extreme values or discontinuities)
        gradient_x = np.abs(np.diff(svf_data, axis=1))
        gradient_y = np.abs(np.diff(svf_data, axis=0))
        
        max_gradient_x = np.max(gradient_x[~np.isnan(gradient_x)])
        max_gradient_y = np.max(gradient_y[~np.isnan(gradient_y)])
        
        print(f"\\nInterpolation quality checks:")
        print(f"  Max gradient X: {max_gradient_x:.3f}")
        print(f"  Max gradient Y: {max_gradient_y:.3f}")
        
        if max_gradient_x > 0.3 or max_gradient_y > 0.3:
            print("  WARNING: Large gradients detected - possible interpolation artifacts")
        else:
            print("  Good: Gradients are reasonable, interpolation looks smooth")
        
        # Cleanup
        for path in [test_dsm_path, svf_path]:
            if os.path.exists(path):
                os.remove(path)
                
        print("\\nLarge dataset test completed successfully!")
        
    except Exception as e:
        print(f"Error in SVF calculation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_large_svf()