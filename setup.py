# -*- coding: utf-8 -*-
import pathlib
from setuptools import find_packages, setup

HERE = pathlib.Path(__file__).parent

VERSION = '1.0.0'
PACKAGE_NAME = 'greenshadow'
AUTHOR = 'Jose Manuel Broto, Gerard Mor, Maite Sellart'
AUTHOR_EMAIL = 'jmbroto@cimne.upc.edu'
URL = 'https://github.com/BeeGroup-cimne/greenshadow'

LICENSE = 'MIT'
DESCRIPTION = 'Python library for environmental shading analysis using LiDAR data and hillshade algorithms with maximum detail for rural and urban areas.'
LONG_DESCRIPTION = (HERE / "README.md").read_text(
    encoding='utf-8')
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
    "affine>=2.3.0",
    "contextily>=1.3.0",
    "geopandas>=0.13.0",
    "laspy[lazrs]>=2.4.0",
    "matplotlib>=3.5.0",
    "numpy>=1.21.0",
    "pandas>=1.5.0",
    "pillow>=9.0.0",
    "pyproj>=3.3.0",
    "rasterio>=1.3.0",
    "requests>=2.28.0",
    "scipy>=1.9.0",
    "shapely>=1.8.0",
    "tqdm>=4.64.0",
    "GDAL>=3.4.0",
]

OPTIONAL_REQUIRES = {
    'solar': ['pysolar>=0.10', 'sunpy>=4.0.0'],
    'pdal': ['pdal>=3.0.0'],
    # zarr/fsspec/aiohttp: hypercadaster's ERA5-Land ARCO Zarr path (else CDS-queue fallback)
    'weather': ['hypercadaster_ES>=1.2.3', 'zarr>=2.18', 'fsspec>=2023.1.0', 'aiohttp>=3.8'],
    'cuda': ['numba-cuda[cu13]'],  # pip CUDA toolkit (libNVVM) for numba
    'dev': ['pytest>=7.0.0', 'black>=22.0.0', 'flake8>=4.0.0'],
    'viz3d': ['plotly>=5.0.0'],
}

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type=LONG_DESC_TYPE,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    url=URL,
    install_requires=INSTALL_REQUIRES,
    extras_require=OPTIONAL_REQUIRES,
    license=LICENSE,
    packages=find_packages(),
    include_package_data=True,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: GIS',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
    ],
    keywords='lidar gis solar shading hillshade remote-sensing urban-planning',
    project_urls={
        'Bug Reports': 'https://github.com/BeeGroup-cimne/greenshadow/issues',
        'Source': 'https://github.com/BeeGroup-cimne/greenshadow',
        'Documentation': 'https://github.com/BeeGroup-cimne/greenshadow/blob/main/README.md',
    }
)
