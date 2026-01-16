#!/usr/bin/env python3
"""
Test script to verify AOI masking in build_dsm_from_lidar_pure function.
"""
import os
import tempfile
import geopandas as gpd
from shapely.geometry import Polygon
import greenshadow as gs

def test_aoi_masking():
    """Test the AOI masking functionality."""
    print("Testing AOI masking functionality...")
    
    # Check if we have a DSM file to test with
    dsm_path = "/home/gmor/Nextcloud2/Beegroup/data/greenshadow/Alpicat/dsm.tif"
    if not os.path.exists(dsm_path):
        print(f"DSM file not found: {dsm_path}")
        print("Please run get_lidar_from_polygon.py first to generate DSM")
        return
    
    # Check if we have LAZ files
    laz_dir = "/home/gmor/Nextcloud2/Beegroup/data/greenshadow/icgc"
    if not os.path.exists(laz_dir):
        print(f"LAZ directory not found: {laz_dir}")
        print("Please run get_lidar_from_polygon.py first to download LAZ files")
        return
    
    # Get some LAZ files for testing
    import glob
    laz_paths = glob.glob(f"{laz_dir}/*.laz")[:2]  # Just use 2 files for quick test
    
    if not laz_paths:
        print(f"No LAZ files found in {laz_dir}")
        return
    
    print(f"Found {len(laz_paths)} LAZ files for testing")
    
    # Create a small test AOI (polygon within the original area)
    # Get the bounds from the existing DSM
    import rasterio
    with rasterio.open(dsm_path) as src:
        bounds = src.bounds
        crs = src.crs
        print(f"Original DSM bounds: {bounds}")
        print(f"Original DSM CRS: {crs}")
        
    # Create a smaller polygon within these bounds for testing
    minx, miny, maxx, maxy = bounds
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    
    # Create a square polygon that's 50% of the original area
    width = (maxx - minx) * 0.25
    height = (maxy - miny) * 0.25
    
    test_polygon = Polygon([
        (center_x - width, center_y - height),
        (center_x + width, center_y - height),
        (center_x + width, center_y + height),
        (center_x - width, center_y + height),
        (center_x - width, center_y - height)
    ])
    
    print(f"Test AOI polygon bounds: {test_polygon.bounds}")
    
    # Create temporary directory for test output
    with tempfile.TemporaryDirectory() as temp_dir:
        test_dsm_path = f"{temp_dir}/test_dsm_with_aoi.tif"
        
        print("Building DSM with AOI masking...")
        try:
            result_path = gs.build_dsm_from_lidar_pure(
                laz_paths=laz_paths,
                out_dsm=test_dsm_path,
                resolution=2.0,  # Use coarser resolution for faster processing
                lidar_epsg=25831,  # UTM Zone 31N for Catalonia
                aoi=test_polygon,
                aoi_epsg=25831,  # Same as LiDAR
                fill_gaps="nearest",
                aggregation="max"
            )
            
            print(f"DSM with AOI masking created: {result_path}")
            
            # Verify the result
            with rasterio.open(result_path) as dst:
                data = dst.read(1)
                nodata_value = dst.nodata
                
                # Count pixels
                total_pixels = data.size
                valid_pixels = np.sum(data != nodata_value)
                nodata_pixels = np.sum(data == nodata_value)
                
                print(f"Result statistics:")
                print(f"  Total pixels: {total_pixels}")
                print(f"  Valid pixels: {valid_pixels}")
                print(f"  NoData pixels: {nodata_pixels}")
                print(f"  Valid pixel percentage: {(valid_pixels/total_pixels)*100:.1f}%")
                
                # The valid pixels should be roughly in the center 25% of the image
                if valid_pixels > 0 and nodata_pixels > 0:
                    print("✅ AOI masking appears to be working - some pixels are masked as nodata")
                else:
                    print("⚠️  Unexpected result - check AOI masking logic")
                    
        except Exception as e:
            print(f"❌ Error testing AOI masking: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    import numpy as np
    test_aoi_masking()