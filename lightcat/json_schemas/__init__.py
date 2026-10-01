import importlib

JSON_SCHEMA_FOLDER_PATH = "lightcat.json_schemas"

CATALOGUES = {
    "default": f"{JSON_SCHEMA_FOLDER_PATH}.default"
}


def get_catalogue(catalogue: str):
    """
    Get catalogue schema by name.
    
    Args:
        catalogue: Name of the catalogue schema to load
        
    Returns:
        BotanicalCatalogue schema class
        
    Raises:
        KeyError: If catalogue not found in CATALOGUES
        ImportError: If module cannot be imported or schema not found
    """
    catalogues = CATALOGUES

    try:
        module_path = catalogues[catalogue]
    except KeyError:
        available = ", ".join(catalogues.keys())
        raise KeyError(f"Catalogue '{catalogue}' not found. Available catalogues: {available}")
    
    try:
        module = importlib.import_module(module_path)
        botanical_catalogue_schema = getattr(module, "BotanicalCatalogue")
        return botanical_catalogue_schema
    except (ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Cannot import catalogue '{catalogue}' from {module_path}.BotanicalCatalogue: {e}")