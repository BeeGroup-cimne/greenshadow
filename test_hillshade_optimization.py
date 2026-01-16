#!/usr/bin/env python3
"""
Test script to verify hillshade optimization works correctly.
"""
import time
import tempfile
import os
import shutil
import greenshadow as gs

def test_hillshade_performance():
    """Test the optimized hillshade generation."""
    print("Testing hillshade performance optimization...")
    
    # Create a small test DSM using an existing one
    dsm_path = "/home/gmor/Nextcloud2/Beegroup/data/greenshadow/Alpicat/dsm.tif"
    if not os.path.exists(dsm_path):
        print(f"DSM file not found: {dsm_path}")
        return
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dir = f"{temp_dir}/test_hillshades"
        
        # Test with just 4 positions for quick test
        azimuth_altitude = [
            (45.0, 30.0),
            (90.0, 30.0),  
            (135.0, 30.0),
            (180.0, 30.0)
        ]
        
        print(f"Testing with {len(azimuth_altitude)} positions...")
        print(f"Output directory: {test_dir}")
        
        # Test single-threaded
        start_time = time.time()
        paths_single = gs.generate_multiple_hillshades_gdal(
            raster_dir=f"{test_dir}/single",
            dsm_path=dsm_path,
            azimuth_altitude=azimuth_altitude,
            max_workers=1,
            qgis_temporal=False,
            include_datetime_info=False
        )
        single_time = time.time() - start_time
        
        print(f"Single-threaded: {single_time:.2f} seconds, {len(paths_single)} files")
        
        # Test multi-threaded
        start_time = time.time()
        paths_multi = gs.generate_multiple_hillshades_gdal(
            raster_dir=f"{test_dir}/multi",
            dsm_path=dsm_path,
            azimuth_altitude=azimuth_altitude,
            max_workers=2,
            qgis_temporal=False,
            include_datetime_info=False
        )
        multi_time = time.time() - start_time
        
        print(f"Multi-threaded (2 workers): {multi_time:.2f} seconds, {len(paths_multi)} files")
        
        if multi_time > 0:
            speedup = single_time / multi_time
            print(f"Speedup: {speedup:.2f}x")
        
        print("Test completed successfully!")

if __name__ == "__main__":
    test_hillshade_performance()