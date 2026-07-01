import os
import re
import glob
from collections import defaultdict
from zipfile import ZipFile
from urllib.parse import unquote
from .utils import _ensure_same_crs, _delete_all_files_in_directory, _fetch_html, _download_file, _unzip_directory, _validate_and_fix_geometries
from .logging_utils import warning, error, info
import requests
import pandas as pd
import geopandas as gpd
from bs4 import BeautifulSoup

SIGPAC_BASE_URL = "https://www.fega.gob.es/atom"
SIGPAC_LAYER = "recinto"

def _get_sigpac_provinces(year: int) -> dict:
    """
    Returns a mapping:
        { '25': '25 - LLEIDA', ... }
    """
    url = f"{SIGPAC_BASE_URL}/{year}/rec_{year}/"
    soup = BeautifulSoup(_fetch_html(url), "html.parser")

    provinces = {}
    for link in soup.find_all("a"):
        href = unquote(link.get("href", "").strip("/"))
        if re.fullmatch(r"\d{2} - .+", href):
            code, _ = href.split(" - ", 1)
            provinces[code] = href

    return provinces


def _parse_latest_sigpac_files(
    html: str,
    year: int,
    municipality_codes: set[int]
) -> dict:
    """
    Returns:
        { municipality_code: {filename, timestamp} }
    """
    soup = BeautifulSoup(html, "html.parser")
    out = {}

    pattern = re.compile(rf"(\d{{5}})_rec_{year}_(\d{{8}})_gpkg\.zip")

    for link in soup.find_all("a"):
        filename = link.get("href", "")
        match = pattern.match(filename)
        if not match:
            continue

        mun_code = int(match.group(1))
        timestamp = match.group(2)

        if mun_code not in municipality_codes:
            continue

        if mun_code not in out or timestamp > out[mun_code]["timestamp"]:
            out[mun_code] = {
                "filename": filename,
                "timestamp": timestamp
            }

    return out


def _download_and_extract_sigpac(
    file_url: str,
    output_dir: str,
    zip_filename: str
) -> str:
    """
    Downloads and extracts a SIGPAC GPKG if not already present.
    Returns the path to the extracted .gpkg file.
    """
    os.makedirs(output_dir, exist_ok=True)

    gpkg_filename = zip_filename.replace("_gpkg.zip", ".gpkg")
    gpkg_path = os.path.join(output_dir, gpkg_filename)

    if os.path.exists(gpkg_path):
        return gpkg_path

    zip_path = os.path.join(output_dir, zip_filename)

    with requests.get(file_url, stream=True) as response:
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(8192):
                f.write(chunk)

    with ZipFile(zip_path) as zip_ref:
        zip_ref.extractall(output_dir)

    os.remove(zip_path)
    return gpkg_path


def _gather_sigpac_gpkgs(
    municipality_codes: list[str],
    year: int,
    workdir: str
) -> list[str]:
    """
    Downloads the SIGPAC GPKG of a specific year per municipality.
    Returns a list of local .gpkg paths.
    """
    municipality_codes = {int(c) for c in municipality_codes}
    provinces = _get_sigpac_provinces(year)

    municipalities_by_province = defaultdict(set)
    for code in municipality_codes:
        municipalities_by_province[str(code).zfill(5)[:2]].add(code)

    gpkg_files = []

    for prov_code, prov_name in provinces.items():
        if prov_code not in municipalities_by_province:
            continue

        province_url = f"{SIGPAC_BASE_URL}/{year}/rec_{year}/{prov_name}"
        info(f"\nProcessing province: {prov_name}")

        html = _fetch_html(province_url)
        latest_files = _parse_latest_sigpac_files(
            html,
            year,
            municipalities_by_province[prov_code]
        )

        for i in latest_files.values():
            file_url = f"{province_url}/{i['filename']}"
            gpkg_files.append(
                _download_and_extract_sigpac(
                    file_url=file_url,
                    output_dir=workdir,
                    zip_filename=i["filename"]
                )
            )

    return gpkg_files


def _extract_translation_dict_from_gpkg(gpkg_path: str, layer_name: str, code_field: str, desc_field: str) -> dict:
    """
    Extract translation dictionary from a GPKG attribute table.
    
    Parameters
    ----------
    gpkg_path : str
        Path to the GPKG file
    layer_name : str 
        Name of the attribute table layer
    code_field : str
        Name of the code field
    desc_field : str
        Name of the description field
        
    Returns
    -------
    dict
        Mapping from code to description
    """
    try:
        df = gpd.read_file(gpkg_path, layer=layer_name)
        return dict(zip(df[code_field], df[desc_field]))
    except Exception:
        return {}


def _get_sigpac_translation_dictionaries() -> dict:
    """
    Returns translation dictionaries for SIGPAC fields from Spanish to English.
    
    Returns
    -------
    dict
        Dictionary with 'uso_sigpac' and 'incidencias' translation mappings
    """
    
    # Land use codes translation (uso_sigpac)
    uso_sigpac_translations = {
        'AG': 'Water courses and surfaces',
        'CA': 'Roads',
        'CF': 'Citrus-Fruit trees association',
        'CI': 'Citrus',
        'CS': 'Citrus-Nuts association',
        'CV': 'Citrus-Vineyard association',
        'ED': 'Buildings',
        'EP': 'Landscape element',
        'FF': 'Fruit trees-Nuts association',
        'FL': 'Nuts and olive trees',
        'FO': 'Forest',
        'FS': 'Nuts',
        'FV': 'Nuts and vineyard',
        'FY': 'Fruit trees',
        'IM': 'Unproductive',
        'IV': 'Greenhouses and crops under plastic',
        'MT': 'Scrubland',
        'OC': 'Olive-Citrus association',
        'OF': 'Olive-Fruit trees',
        'OP': 'Other permanent crops',
        'OV': 'Olive trees',
        'PA': 'Pasture with trees',
        'PR': 'Shrub pasture',
        'PS': 'Grassland',
        'TA': 'Arable land',
        'TH': 'Vegetable garden',
        'VF': 'Vineyard-Fruit trees',
        'VI': 'Vineyard',
        'VO': 'Vineyard-Olive trees',
        'ZC': 'Concentrated zone not included in orthophoto',
        'ZU': 'Urban zone',
        'ZV': 'Censored zone'
    }
    
    # Common incidence codes (this would be a subset, full mapping would be very long)
    incidencias_translations = {
        '7': 'Use assigned by cadastral information',
        '11': 'Scattered trees',
        '12': 'Contains other uses without subdivision',
        '13': 'Crop use observed in orthophoto',
        '14': 'Crop surface extending into road or river network',
        '15': 'Parcel included in forestry certificate graphic information',
        '17': 'Parcel included in vineyard registry - vine removal',
        '19': 'Citrus according to citrus GIS',
        '20': 'Vineyard according to vineyard GIS',
        '21': 'Rice cadastral',
        '37': 'Parcel included in forestry certificate alphanumeric information',
        '39': 'Citrus surface does not occupy entire plot due to association',
        '46': 'Non-arable use parcel in declared parcels file',
        '47': 'Plowed surface observed in orthophoto',
        '49': 'Isolated or framework trees in combined arable use',
        '58': 'Vineyard according to vineyard registry in part of SIGPAC parcel',
        '62': 'Pasture use in permanent pasture database',
        '63': 'Non-pasture use in permanent pasture database',
        '74': 'SIGPAC use information provided by autonomous community',
        '75': 'Farming system information provided by autonomous community',
        '80': 'Data from autonomous community information',
        '81': 'Energy installations',
        '83': 'Greenhouse tomato',
        '96': 'SIGPAC use validated by remote sensing control 2017',
        '97': 'SIGPAC use studied by remote sensing control 2017',
        '98': 'SIGPAC use validated by remote sensing control 2018',
        '99': 'SIGPAC use studied by remote sensing control 2018',
        '100': 'SIGPAC use validated by remote sensing control 2019',
        '111': 'Vineyard-vegetables association',
        '112': 'Banana plantations',
        '116': 'Contains landscape elements',
        '117': 'Non-subsidizable: Abandoned crop',
        '119': 'Urban cadastral',
        '120': 'Measured steppe zone',
        '124': 'Short rotation forest surfaces',
        '127': 'Verified irrigation. No concession exists',
        '128': 'Uncultivated plots with environmental values or agro-environmental measures',
        '129': 'Truffle plantation or mycorrhized oaks',
        '130': 'Regular forest plantations',
        '132': 'Rocky area according to SIOSE coverage',
        '133': 'Entire plot is undefined landscape element',
        '134': 'Landscape element - trees in row',
        '135': 'Landscape element - trees in groups up to 0.3 ha',
        '136': 'Landscape element - boundaries up to 10 m width',
        '137': 'Landscape element - ponds, lagoons, small wetlands up to 0.1 ha',
        '138': 'Landscape element - natural vegetation or rock islands up to 0.1 ha',
        '139': 'Landscape element - terraces up to 10 m width',
        '140': 'Subsidizability coefficient information from autonomous community',
        '142': 'Asparagus crop',
        '143': 'Scrubland undergrowth',
        '144': 'Wasteland',
        '145': 'Woody aromatic plants',
        '146': 'CAP manual assignment small plots',
        '148': 'Parcel belonging to a Tancat',
        '149': 'Transformed sensitive pasture plot',
        '150': 'Plot with banana and other tropical fruit mix',
        '151': 'Fruit trees under plastic',
        '152': 'Unproductive according to MFE coverage',
        '153': 'CAP manual update with 2016 coefficient',
        '154': 'Field verified 2017',
        '155': 'Nuts not meeting minimum density - field/administrative control',
        '156': 'Pasture control over 10 years',
        '157': 'SIGPAC use change from forest/pasture without environmental authorization',
        '158': 'Fallow for 5 consecutive years',
        '159': 'Plot subsidizability conditioned by fire',
        '160': 'Contains stone walls (balates)',
        '161': 'Nuts not meeting minimum density - registered tree count',
        '162': 'Field verified 2018',
        '163': 'Use change informed as exempt from authorization',
        '164': 'Use change informed as authorized',
        '165': 'SIGPAC woody use change in LIC/ZEPA without authorization',
        '166': 'Use change informed as not authorized',
        '167': 'Pine inoculated with milk cap mushrooms',
        '168': 'Woody pastures without herbaceous majority',
        '169': 'Pine for pine nut production',
        '170': 'Field verified 2019',
        '171': 'SIGPAC use studied by remote sensing control 2019',
        '173': 'Agricultural activity verified 2016',
        '174': 'Agricultural activity verified 2017',
        '175': 'Agricultural activity verified 2018',
        '176': 'Agricultural activity verified 2019',
        '177': 'Permanent crops with maintenance activity 5 consecutive years',
        '178': 'Monitoring 2019: Pending confirmation by orthophoto renewal',
        '179': 'Plot with cultivation rights in public domain',
        '180': 'SIGPAC use validated by remote sensing control 2020',
        '181': 'SIGPAC use studied by remote sensing control 2020',
        '182': 'Field verified 2020',
        '183': 'Monitoring 2020: Pending confirmation by orthophoto renewal',
        '184': 'Soilless cultivation',
        '185': 'Agricultural activity verified 2020',
        '186': 'Permanent pasture with maintenance activity 5 consecutive years',
        '187': 'Pending control due to covid-19',
        '188': 'SIGPAC use validated by remote sensing control 2021',
        '189': 'SIGPAC use studied by remote sensing control 2021',
        '190': 'Field verified 2021',
        '191': 'Monitoring 2021: Pending confirmation by orthophoto renewal',
        '192': 'Agricultural activity verified 2021',
        '193': 'Public domain according to cadastre',
        '196': 'Field verified 2022',
        '197': 'Monitoring 2022: Pending confirmation by orthophoto renewal',
        '198': 'Agricultural activity verified 2022',
        '199': 'Inactive plot',
        '202': 'Parcel concentration region information from autonomous community',
        '204': 'Livestock route information from autonomous community',
        '208': 'External CSP pending raster change',
        '209': 'Reviewed by photointerpretation by autonomous community',
        '210': 'Plots affected by 2021 eruption',
        '211': 'Pasture without agricultural activity',
        '213': 'Meadow suitable for mowing with altitude <= 300 m',
        '214': 'Meadow suitable for mowing with altitude > 300 m',
        '215': 'Landscape element - traditional constructions',
        '216': 'Landscape element - hedge up to 10 m width',
        '217': 'Landscape element - isolated trees',
        '218': 'Field verified 2023',
        '219': 'Monitoring 2023: Pending confirmation by orthophoto renewal',
        '220': 'Agricultural activity verified 2023',
        '221': 'Irrigated farmland exploited in last 3 years',
        '222': 'Landscape element - stone wall',
        '223': 'Permanent crop headland',
        '224': 'Woody crop with terraces',
        '225': 'Plot with CSP blocked after fire',
        '226': 'Mushroom cultivation',
        '227': 'Meadow suitable for mowing',
        '228': 'Field verified 2024',
        '229': 'Monitoring 2024: Pending confirmation by orthophoto renewal',
        '230': 'Agricultural activity verified 2024',
        '231': 'Availability not accredited',
        '233': 'Field verified 2025',
        '234': 'Monitoring 2025: Pending confirmation by orthophoto renewal',
        '235': 'Agricultural activity verified 2025'
    }
    
    return {
        'uso_sigpac': uso_sigpac_translations,
        'incidencias': incidencias_translations
    }


def harmonize_sigpac_fields(
    parcels_gdf: gpd.GeoDataFrame, 
    gpkg_path: str | None = None,
    use_builtin_translations: bool = True
) -> gpd.GeoDataFrame:
    """
    Harmonizes SIGPAC field acronyms by adding English translation columns.
    
    Parameters
    ----------
    parcels_gdf : GeoDataFrame
        SIGPAC parcels dataframe
    gpkg_path : str, optional
        Path to a SIGPAC GPKG file to extract translations from attribute tables.
        If None, uses the source_gpkg field from parcels to find a representative file.
    use_builtin_translations : bool, default True
        Whether to use built-in translation dictionaries as fallback
        
    Returns
    -------
    GeoDataFrame
        Parcels dataframe with additional English translation columns:
        - uso_sigpac_en: English translation of uso_sigpac
        - incidencias_en: English translation of incidencias
    """
    parcels = parcels_gdf.copy()
    
    # Get translation dictionaries
    if use_builtin_translations:
        translations = _get_sigpac_translation_dictionaries()
    else:
        translations = {'uso_sigpac': {}, 'incidencias': {}}
    
    # Try to extract translations from GPKG if path provided or available
    if gpkg_path is None and 'source_gpkg' in parcels.columns:
        # Use first available source GPKG
        source_gpkgs = parcels['source_gpkg'].dropna().unique()
        if len(source_gpkgs) > 0:
            # Try to reconstruct the full path (this is a heuristic)
            gpkg_filename = source_gpkgs[0]
            # Look in common directories or use relative path
            possible_paths = [
                gpkg_filename,
                f"/media/gmor/data/greenshadow/*/rural_parcels/{gpkg_filename}",
                f"./rural_parcels/{gpkg_filename}"
            ]
            for path_pattern in possible_paths:
                if '*' in path_pattern:
                    matches = glob.glob(path_pattern)
                    if matches:
                        gpkg_path = matches[0]
                        break
                elif os.path.exists(path_pattern):
                    gpkg_path = path_pattern
                    break
    
    if gpkg_path and os.path.exists(gpkg_path):
        # Extract translations from GPKG attribute tables
        uso_dict = _extract_translation_dict_from_gpkg(gpkg_path, 'cod_usosigpac', 'cod_uso', 'descripcion')
        incidencias_dict = _extract_translation_dict_from_gpkg(gpkg_path, 'cod_incidencia', 'cod_incidencia', 'descripcion')
        
        # Merge with built-in translations (GPKG takes priority)
        if uso_dict:
            # Create English translations for Spanish descriptions
            uso_dict_en = {}
            for code, spanish_desc in uso_dict.items():
                if code in translations['uso_sigpac']:
                    uso_dict_en[code] = translations['uso_sigpac'][code]
                else:
                    # Keep Spanish description if no English translation available
                    uso_dict_en[code] = spanish_desc
            translations['uso_sigpac'].update(uso_dict_en)
        
        if incidencias_dict:
            # For incidencias, use numeric keys
            incidencias_dict_en = {}
            for key, spanish_desc in incidencias_dict.items():
                str_key = str(key)
                if str_key in translations['incidencias']:
                    incidencias_dict_en[str_key] = translations['incidencias'][str_key]
                else:
                    # Keep Spanish description if no English translation available
                    incidencias_dict_en[str_key] = spanish_desc
            translations['incidencias'].update(incidencias_dict_en)
    
    # Apply translations
    if 'uso_sigpac' in parcels.columns:
        parcels['uso_sigpac_en'] = parcels['uso_sigpac'].map(translations['uso_sigpac'])
        # Fill missing values with original code
        parcels['uso_sigpac_en'] = parcels['uso_sigpac_en'].fillna(parcels['uso_sigpac'])
    
    if 'incidencias' in parcels.columns:
        # Handle multiple incidencias separated by commas
        def translate_incidencias(incidencias_str):
            if pd.isna(incidencias_str) or incidencias_str == '':
                return None
            
            codes = [code.strip() for code in str(incidencias_str).split(',')]
            translations_list = []
            for code in codes:
                if code in translations['incidencias']:
                    translations_list.append(translations['incidencias'][code])
                else:
                    translations_list.append(code)  # Keep original if no translation
            
            return ', '.join(translations_list)
        
        parcels['incidencias_en'] = parcels['incidencias'].apply(translate_incidencias)
    
    return parcels


def load_spanish_sigpac_parcels(
    municipality_codes: list[str],
    year: int,
    workdir: str,
    target_crs: str | None = None,
    harmonize_fields: bool = False
) -> gpd.GeoDataFrame:
    """
    Builds and returns a national GeoDataFrame with SIGPAC recintos.

    Parameters
    ----------
    municipality_codes : list[str]
        List of 5-digit municipality codes
    year : int
        Year of SIGPAC data
    workdir : str
        Working directory for downloaded files
    target_crs : str, optional
        Target CRS for the output
    harmonize_fields : bool, default False
        Whether to add English translations for SIGPAC field acronyms

    Returns
    -------
    GeoDataFrame
        SIGPAC parcels with optional English field translations
    """
    if isinstance(municipality_codes,str):
        municipality_codes = [municipality_codes]

    gpkg_files = _gather_sigpac_gpkgs(
        municipality_codes=municipality_codes,
        year=year,
        workdir=workdir
    )

    gdfs = []

    for gpkg in gpkg_files:
        try:
            gdf = gpd.read_file(gpkg, layer=SIGPAC_LAYER)
            if len(gdf) == 0:
                warning(f"Empty layer in {gpkg}")
                continue

            gdf["source_gpkg"] = os.path.basename(gpkg)
            gdfs.append(gdf)

        except Exception as e:
            warning(f"Failed to read {gpkg}: {e}")

    if not gdfs:
        raise error(
            "No SIGPAC parcels were loaded.\n"
            "Check:\n"
            " - municipality_codes\n"
            " - year\n"
            " - FEGA availability\n"
            " - layer name ('recinto')"
        )

    parcels_sigpac_spain = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True),
        geometry="geometry",
        crs=gdfs[0].crs
    )

    if target_crs:
        parcels_sigpac_spain = parcels_sigpac_spain.to_crs(target_crs)

    parcels_sigpac_spain["area_ha"] = (
        parcels_sigpac_spain.to_crs("EPSG:25831").area / 10_000
    )

    # Add English field translations if requested
    if harmonize_fields:
        parcels_sigpac_spain = harmonize_sigpac_fields(parcels_sigpac_spain)

    return parcels_sigpac_spain
def load_spanish_preserved_regions(
    data_dir: str,
    datasets: list[str] = ("natura2000", "enp"),
    *,
    join_geometries_per_resource: bool = True,
    update: bool = False
) -> gpd.GeoDataFrame:
    """
    Load preserved natural areas of Spain (Natura 2000, ENP).

    Data are downloaded, cached, optionally dissolved by resource,
    and returned as a GeoDataFrame.

    Parameters
    ----------
    data_dir : str
        Base directory for cached datasets.
    datasets : list[str]
        Preserved datasets to load. Allowed values: "natura2000", "enp".
    join_geometries_per_resource : bool
        If True, dissolve geometries by resource.
    update : bool
        Force re-download of datasets.

    Returns
    -------
    GeoDataFrame
        Preserved regions with at least columns:
        - resource
        - geometry
    """

    os.makedirs(data_dir, exist_ok=True)
    datasets = sorted(datasets)

    out_file = (
        f"{data_dir}/all_{'~'.join(datasets)}"
        f"_joined{join_geometries_per_resource}.gpkg"
    )

    if os.path.exists(out_file):
        return gpd.read_file(out_file)

    allowed = {"natura2000", "enp"}
    if not any(d in allowed for d in datasets):
        raise ValueError(
            f"No valid preserved datasets requested. "
            f"Allowed datasets: {', '.join(sorted(allowed))}"
        )

    gdfs: list[gpd.GeoDataFrame] = []

    # ------------------------------------------------------------------
    # Natura 2000
    # ------------------------------------------------------------------
    if "natura2000" in datasets:

        base = f"{data_dir}/n2000"
        zip_dir = f"{base}/zip"
        unzip_dir = f"{base}/unzip"

        os.makedirs(zip_dir, exist_ok=True)
        os.makedirs(unzip_dir, exist_ok=True)

        if update or not os.listdir(zip_dir):
            _delete_all_files_in_directory(zip_dir)
            _delete_all_files_in_directory(unzip_dir)

            _download_file(
                dir=zip_dir,
                url=(
                    "https://www.miteco.gob.es/content/dam/miteco/es/"
                    "biodiversidad/servicios/banco-datos-naturaleza/"
                    "3-rn2000/rn2000-geojson.zip"
                ),
            )
            _unzip_directory(zip_directory=zip_dir, unzip_directory=unzip_dir)

        try:
            available_files = os.listdir(unzip_dir)
            info(f"Looking for Natura 2000 file containing 'Medalpatl' in: {available_files}")
            
            # Try different patterns that might exist in the files
            patterns_to_try = ["Medalpatl", "medalpatl", "MEDALPATL", "Red_Natura", "natura", "n2000"]
            src_file = None
            
            for pattern in patterns_to_try:
                try:
                    src_file = next(f for f in available_files if pattern in f)
                    info(f"Found Natura 2000 file using pattern '{pattern}': {src_file}")
                    break
                except StopIteration:
                    continue
            
            if src_file is None:
                raise StopIteration
                
        except StopIteration:
            available_files = os.listdir(unzip_dir)
            zip_files = os.listdir(zip_dir) if os.path.exists(zip_dir) else []
            raise FileNotFoundError(
                f"No file containing expected patterns found in {unzip_dir}. "
                f"Available files in unzip dir: {available_files}. "
                f"Available files in zip dir: {zip_files}. "
                f"Try setting update=True to re-download the data."
            )
        n2000 = gpd.read_file(f"{unzip_dir}/{src_file}")

        n2000 = (
            n2000
            .drop(columns=["AC"], errors="ignore")
            .rename(
                columns={
                    "SITE_CODE": "id",
                    "SITE_NAME": "name",
                    "TIPO": "type",
                    "HECTAREAS": "area",
                }
            )
        )

        n2000["resource"] = "Red Natura 2000"
        n2000["area"] = n2000["area"].astype(float) * 10_000

        gdfs.append(n2000)

    # ------------------------------------------------------------------
    # Espacios Naturales Protegidos
    # ------------------------------------------------------------------
    if "enp" in datasets:

        base = f"{data_dir}/enp"
        zip_dir = f"{base}/zip"
        unzip_dir = f"{base}/unzip"

        os.makedirs(zip_dir, exist_ok=True)
        os.makedirs(unzip_dir, exist_ok=True)

        if update or not os.listdir(zip_dir):
            _delete_all_files_in_directory(zip_dir)
            _delete_all_files_in_directory(unzip_dir)

            _download_file(
                dir=zip_dir,
                url=(
                    "https://www.miteco.gob.es/content/dam/miteco/es/"
                    "biodiversidad/servicios/banco-datos-naturaleza/"
                    "enp/enp-geojson.zip"
                ),
            )
            _unzip_directory(zip_directory=zip_dir, unzip_directory=unzip_dir)

        try:
            available_files = os.listdir(unzip_dir)
            info(f"Looking for ENP file containing 'PB' in: {available_files}")
            
            # Try different patterns that might exist in the files
            patterns_to_try = ["PB", "pb", "ENP", "enp", "espacios", "protegidos"]
            src_file = None
            
            for pattern in patterns_to_try:
                try:
                    src_file = next(f for f in available_files if pattern in f)
                    info(f"Found ENP file using pattern '{pattern}': {src_file}")
                    break
                except StopIteration:
                    continue
            
            if src_file is None:
                raise StopIteration
                
        except StopIteration:
            available_files = os.listdir(unzip_dir)
            zip_files = os.listdir(zip_dir) if os.path.exists(zip_dir) else []
            raise FileNotFoundError(
                f"No file containing expected patterns found in {unzip_dir}. "
                f"Available files in unzip dir: {available_files}. "
                f"Available files in zip dir: {zip_files}. "
                f"Try setting update=True to re-download the data."
            )
        enp = gpd.read_file(f"{unzip_dir}/{src_file}")

        enp = (
            enp
            .drop(columns=["NUT2", "DESIG_ABBR", "SITE_CDDA"], errors="ignore")
            .rename(
                columns={
                    "SITE_CODE_": "id",
                    "SITE_NAME": "name",
                    "ODESIGNATE": "type",
                    "Sup_ha": "area",
                }
            )
        )

        enp["resource"] = "Espacios Naturales Protegidos"
        enp["area"] = enp["area"].astype(float) * 10_000

        # Align schema / CRS with Natura 2000 if loaded
        if gdfs:
            # Only keep columns that exist in both datasets
            common_columns = [col for col in gdfs[0].columns if col in enp.columns]
            enp = enp[common_columns]
            enp = enp.to_crs(gdfs[0].crs)

        gdfs.append(enp)

    # ------------------------------------------------------------------
    # Merge, dissolve, validate
    # ------------------------------------------------------------------
    gdf = gpd.GeoDataFrame(
        pd.concat(gdfs, ignore_index=True),
        geometry="geometry",
        crs=gdfs[0].crs,
    )

    if join_geometries_per_resource:
        gdf = (
            gdf
            .dissolve(by="resource")
            .reset_index()
            .drop(columns=["id", "name", "type", "area"], errors="ignore")
        )

    gdf = _validate_and_fix_geometries(gdf)

    gdf.to_file(out_file, driver="GPKG")
    return gdf



def tag_parcels_with_preserved_areas(
    parcels_gdf: gpd.GeoDataFrame,
    preserved_gdf: gpd.GeoDataFrame,
    *,
    area_crs: str = "EPSG:25831",
    min_intersection_ratio: float = 0.0
) -> gpd.GeoDataFrame:
    """
    Adds preserved-area information to each parcel.

    Parameters
    ----------
    parcels_gdf : GeoDataFrame
        Parcel geometries (e.g. SIGPAC recintos).
    preserved_gdf : GeoDataFrame
        Preserved regions (Natura 2000, ENP), optionally dissolved.
    area_crs : str
        CRS used for area computations.
    min_intersection_ratio : float
        Minimum fraction of parcel area to be considered preserved
        (e.g. 0.05 = at least 5%).

    Returns
    -------
    GeoDataFrame
        parcels_gdf with added columns:
        - is_preserved
        - preserved_resource
        - preserved_area_m2
        - preserved_area_ratio
    """

    parcels = parcels_gdf.copy()
    preserved = preserved_gdf.copy()

    parcels, preserved = _ensure_same_crs(parcels, preserved)

    # --- Spatial intersection ---
    inter = gpd.overlay(
        parcels,
        preserved[["resource", "geometry"]],
        how="intersection"
    )

    if inter.empty:
        parcels["is_preserved"] = False
        parcels["preserved_resource"] = None
        parcels["preserved_area_m2"] = 0.0
        parcels["preserved_area_ratio"] = 0.0
        return parcels

    # --- Area computations ---
    inter_area = (
        inter.to_crs(area_crs)
        .geometry
        .area
    )
    inter["preserved_area_m2"] = inter_area

    parcel_area = (
        parcels.to_crs(area_crs)
        .geometry
        .area
    )
    parcels["_parcel_area_m2"] = parcel_area

    # Aggregate per parcel
    agg = (
        inter
        .groupby(inter.index)
        .agg(
            preserved_area_m2=("preserved_area_m2", "sum"),
            preserved_resource=("resource", lambda x: ", ".join(sorted(set(x))))
        )
    )

    parcels = parcels.join(agg, how="left")
    parcels["preserved_area_m2"] = parcels["preserved_area_m2"].fillna(0.0)
    parcels["preserved_resource"] = parcels["preserved_resource"].fillna(None)

    parcels["preserved_area_ratio"] = (
        parcels["preserved_area_m2"] / parcels["_parcel_area_m2"]
    )

    parcels["is_preserved"] = (
        parcels["preserved_area_ratio"] > min_intersection_ratio
    )

    parcels = parcels.drop(columns="_parcel_area_m2")

    return parcels
