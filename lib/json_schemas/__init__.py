import importlib

CATALOGUES = {
    "default": "lib.json_schemas.default"
}


def get_catalogue(catalogue: str):

    catalogues = CATALOGUES

    try:
        module = importlib.import_module(catalogues[catalogue])
        botanical_catalogue_schema = getattr(module, "BotanicalCatalogue")
        return botanical_catalogue_schema
    except:
        raise ImportError(f"{catalogue} cannot be imported. Provide the correct type of catalogue. \n==> {catalogues.keys()}")