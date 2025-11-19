from pydantic import BaseModel, Field, create_model
import pandas as pd

def create_model_from_dict(model_name: str, fields_dict: dict[str, str]) -> type[BaseModel]:
    """
    Create a Pydantic model dynamically from a dictionary.

    Args:
        model_name: Name for the generated model
        fields_dict: Dictionary with field names as keys and descriptions as values

    Returns:
        A dynamically created Pydantic model class
    """
    field_definitions = {
        field_name: (str, Field(..., description=description))
        for field_name, description in fields_dict.items()
    }

    return create_model(model_name, **field_definitions)

def flatten_pydantic(model: BaseModel, sep: str = '.') -> dict:
    model_dict = model.model_dump()
    [flat_dict] = pd.json_normalize(model_dict, sep=sep).to_dict(orient='records')
    return flat_dict