#!/usr/bin/env python3
"""
Test script for Sky View Factor calculation
"""
import os
import greenshadow as gs

def test_sky_view_factor():
    """Test the Sky View Factor calculation function."""
    
    # Check if we have a DSM file to work with
    dsm_path = "/home/gmor/Nextcloud2/Beegroup/data/greenshadow/Alpicat/dsm.tif"
    
    if not os.path.exists(dsm_path):
        print(f"DSM file not found: {dsm_path}")
        print("Please run the get_lidar_from_polygon.py script first to generate a DSM")
        return
    
    print("Testing Sky View Factor calculation...")
    
    # Set output path
    output_dir = "/home/gmor/Nextcloud2/Beegroup/data/greenshadow/Alpicat"
    svf_path = f"{output_dir}/sky_view_factor.tif"
    
    try:
        # Calculate Sky View Factor with different settings for testing
        print("\n=== Quick SVF (coarse resolution) ===")
        svf_quick = gs.calculate_sky_view_factor(
            dsm_path=dsm_path,
            out_path=f"{output_dir}/svf_quick.tif",
            search_radius=50.0,      # Smaller radius for speed
            angular_resolution=10.0,  # Coarser angular resolution
            max_height_angle=90.0
        )
        
        print("\n=== Standard SVF (balanced) ===")
        svf_standard = gs.calculate_sky_view_factor(
            dsm_path=dsm_path,
            out_path=svf_path,
            search_radius=100.0,     # Standard radius
            angular_resolution=5.0,  # Standard angular resolution
            max_height_angle=90.0
        )
        
        print(f"\nSky View Factor files created:")
        print(f"  Quick:    {svf_quick}")
        print(f"  Standard: {svf_standard}")
        
        print("\nTips for using SVF results:")
        print("1. Load the SVF raster in QGIS")
        print("2. Use a color ramp from blue (low SVF) to yellow (high SVF)")
        print("3. SVF values:")
        print("   - 0.0-0.3: Deep urban canyons, courtyards")
        print("   - 0.3-0.6: Street level in urban areas")  
        print("   - 0.6-0.8: Suburban areas, parks")
        print("   - 0.8-1.0: Open areas, rooftops")
        print("4. Combine with solar analysis for thermal comfort studies")
        
    except Exception as e:
        print(f"Error calculating Sky View Factor: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sky_view_factor()