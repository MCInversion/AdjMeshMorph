import pandas as pd

def load_material_properties(filename="materials.csv"):
    """
    Load material properties from a CSV file.

    :param filename: Path to the CSV file.
    :return: DataFrame with material properties.
    """
    return pd.read_csv(filename)

def get_material_properties(material, material_properties):
    """
    Get the properties of a specific material.

    :param material: Name of the material.
    :param material_properties: DataFrame with material properties.
    :return: Young's modulus and Poisson's ratio of the material.
    """
    properties = material_properties[material_properties['Material'] == material]
    if not properties.empty:
        E = properties.iloc[0]['Young\'s Modulus (Pa)']
        nu = properties.iloc[0]['Poisson\'s Ratio']
        return E, nu
    else:
        raise ValueError(f"Material '{material}' not found in properties file.")