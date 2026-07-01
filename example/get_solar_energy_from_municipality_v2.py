# -*- coding: utf-8 -*-
"""
Example: Enhanced roof plane detection using pystruct3d integration

This example demonstrates the improved roof plane detection using the
buildings_infer_v2 module with pystruct3d integration.

Key improvements over the standard detect_roof_planes:
- Optimal horizontal aligned bounding box fitting for better roof plane orientation
- Convex hull-based plane segmentation
- Enhanced RANSAC with bounding box constraints
- Better handling of pitched roofs and complex geometries
"""

import os
import geopandas as gpd
from shapely.geometry import shape
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import greenshadow as gs

# ================================================================
# Configuration
# ================================================================

# Municipality configuration
MUNICIPALITY_NAME = "barcelona"
MUNICIPALITY_CODE = "080193"

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / MUNICIPALITY_NAME
OUTPUT_DIR = BASE_DIR / "output" / MUNICIPALITY_NAME

# Create directories
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================
# Helper functions
# ================================================================


def setup_paths():
    """Setup required file paths for the example."""
    # In a real scenario, these would be paths to actual data files
    # This is a template showing the required inputs

    paths = {
        "dsm_path": str(DATA_DIR / "dsm.tif"),
        "slope_path": str(DATA_DIR / "slope.tif"),
        "aspect_path": str(DATA_DIR / "aspect.tif"),
        "dsm_class_path": str(DATA_DIR / "classification.tif"),
        "dem_path": str(DATA_DIR / "dem.tif"),
        "output_path": str(OUTPUT_DIR / "roof_planes_v2.gpkg"),
    }

    return paths


def visualize_results(gdf_planes, title="Enhanced Roof Planes (pystruct3d)"):
    """Visualize detected roof planes."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # 1. Roof plane footprints
    ax = axes[0, 0]
    gdf_planes.plot(column="tilt_deg", ax=ax, cmap="viridis",
                    legend=True, legend_kwds={'label': "Tilt (°)"})
    ax.set_title("Roof Planes by Tilt")
    ax.set_aspect('equal')

    # 2. Roof plane azimuth
    ax = axes[0, 1]
    gdf_planes.plot(column="azimuth_deg", ax=ax, cmap="hsv",
                    legend=True, legend_kwds={'label': "Azimuth (°)"})
    ax.set_title("Roof Planes by Azimuth")
    ax.set_aspect('equal')

    # 3. Planarity (RMSE)
    ax = axes[1, 0]
    gdf_planes.plot(column="planarity_rmse", ax=ax, cmap="YlOrRd",
                    legend=True, legend_kwds={'label': "RMSE (m)"})
    ax.set_title("Planarity Quality (RMSE)")
    ax.set_aspect('equal')

    # 4. Detection method
    ax = axes[1, 1]
    if "detection_method" in gdf_planes.columns:
        gdf_planes.plot(column="detection_method", ax=ax, cmap="Set1",
                        categorical=True, legend=True)
        ax.set_title("Detection Method")
    else:
        gdf_planes.plot(ax=ax)
        ax.set_title("All Roof Planes")
    ax.set_aspect('equal')

    plt.suptitle(title, fontsize=16, y=0.995)
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{title.replace(' ', '_')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")

    return fig


def compare_with_standard(v2_results, standard_results=None):
    """Compare v2 results with standard detection."""
    if standard_results is None:
        print("Standard detection results not provided for comparison.")
        return None

    print("\n" + "="*60)
    print("COMPARISON: pystruct3d-enhanced vs Standard Detection")
    print("="*60)

    v2_count = len(v2_results)
    std_count = len(standard_results)

    print(f"\nNumber of roof planes detected:")
    print(f"  Standard: {std_count}")
    print(f"  Enhanced (v2): {v2_count}")
    print(f"  Difference: {v2_count - std_count}")

    # Quality metrics comparison
    print(f"\nPlanarity Quality (RMSE):")
    print(f"  Standard - Mean: {standard_results['planarity_rmse'].mean():.4f} m")
    print(f"  Enhanced - Mean: {v2_results['planarity_rmse'].mean():.4f} m")

    print(f"\nRoughness (MAD/RMSE):")
    print(f"  Standard - Mean: {standard_results['roughness'].mean():.4f}")
    print(f"  Enhanced - Mean: {v2_results['roughness'].mean():.4f}")

    return {
        "v2_count": v2_count,
        "std_count": std_count,
        "v2_rmse_mean": v2_results['planarity_rmse'].mean(),
        "std_rmse_mean": standard_results['planarity_rmse'].mean(),
    }


def main():
    """Main example function."""
    print("="*60)
    print("Enhanced Roof Plane Detection with pystruct3d")
    print("="*60)

    # Setup paths
    paths = setup_paths()

    # Check if pystruct3d is available
    try:
        import pystruct3d
        print("\n✓ pystruct3d is available for enhanced processing")
    except ImportError:
        print("\n✗ pystruct3d not available")
        print("  Install with: pip install git+https://github.com/humantecheu/pystruct3d.git")
        print("  The function will automatically fall back to standard method.")

    # ================================================================
    # Example 1: Basic usage with pystruct3d enabled
    # ================================================================

    print("\n" + "-"*60)
    print("Example 1: Basic enhanced detection")
    print("-"*60)

    """
    # Uncomment when data files are available:
    gdf_planes_v2 = gs.detect_roof_planes_v2(
        dsm_path=paths["dsm_path"],
        slope_path=paths["slope_path"],
        aspect_path=paths["aspect_path"],
        dsm_class_path=paths["dsm_class_path"],
        dem_path=paths["dem_path"],
        output_path=paths["output_path"],
        # pystruct3d-specific parameters
        use_pystruct3d_orientation=True,
        bbox_fitting_method="horizontal_aligned",
        bbox_constraint_threshold=0.1,
        refine_with_3d_points=True,
        # Standard parameters
        min_height_m=1.5,
        min_roof_plane_area=5,
        ridge_splitting_roof_planes=True,
        preclustering_roof_planes=True,
    )

    print(f"\n✓ Detected {len(gdf_planes_v2)} roof planes")
    print(f"✓ Results saved to {paths['output_path']}")

    # ================================================================
    # Example 2: Compare with standard detection
    # ================================================================

    print("\n" + "-"*60)
    print("Example 2: Comparison with standard detection")
    print("-"*60)

    gdf_planes_std = gs.detect_roof_planes(
        dsm_path=paths["dsm_path"],
        slope_path=paths["slope_path"],
        aspect_path=paths["aspect_path"],
        dsm_class_path=paths["dsm_class_path"],
        dem_path=paths["dem_path"],
        output_path=str(OUTPUT_DIR / "roof_planes_standard.gpkg"),
    )

    comparison = compare_with_standard(gdf_planes_v2, gdf_planes_std)

    # ================================================================
    # Example 3: Visualization
    # ================================================================

    print("\n" + "-"*60)
    print("Example 3: Visualization")
    print("-"*60)

    fig = visualize_results(gdf_planes_v2)
    """

    # ================================================================
    # Example 4: Using different pystruct3d settings
    # ================================================================

    print("\n" + "-"*60)
    print("Example 4: Different configuration options")
    print("-"*60)

    print("\nAvailable pystruct3d-enhanced parameters:")
    print("  use_pystruct3d_orientation: bool (default=True)")
    print("    - Enable/disable pystruct3d orientation detection")
    print("\n  bbox_fitting_method: str (default='horizontal_aligned')")
    print("    - 'horizontal_aligned': Optimal orientation via convex hull")
    print("    - 'axis_aligned': Simpler axis-aligned fitting")
    print("\n  bbox_constraint_threshold: float (default=0.1)")
    print("    - Threshold for bbox-based plane assignment")
    print("\n  refine_with_3d_points: bool (default=True)")
    print("    - Use 3D LiDAR points for bbox fitting")

    print("\nExample usage with axis-aligned method:")
    """
    gdf_planes_axis_aligned = gs.detect_roof_planes_v2(
        dsm_path=paths["dsm_path"],
        slope_path=paths["slope_path"],
        aspect_path=paths["aspect_path"],
        dsm_class_path=paths["dsm_class_path"],
        output_path=str(OUTPUT_DIR / "roof_planes_axis_aligned.gpkg"),
        use_pystruct3d_orientation=True,
        bbox_fitting_method="axis_aligned",  # Use axis-aligned fitting
    )
    """

    print("\n" + "="*60)
    print("Example setup complete!")
    print("="*60)
    print("\nTo run the actual detection:")
    print("1. Ensure required data files are available")
    print("2. Uncomment the example code blocks")
    print("3. Run the script: python example/get_solar_energy_from_municipality_v2.py")


if __name__ == "__main__":
    main()