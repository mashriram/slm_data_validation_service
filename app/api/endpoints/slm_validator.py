import logging
from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException

from app.services.slm_validator import SLMValidator
from app.utils.exceptions import DataValidationError

logger = logging.getLogger(__name__)
router = APIRouter()
slm_validator = SLMValidator()


@router.post("/validate-slm-data", response_model=List[Dict[str, Any]])
async def validate_slm_data(data: List[Dict[str, Any]]):
    """
    Receives data from the SLM creation service, validates it,
    and returns it in the proper format for fine-tuning.
    """
    try:
        # The SLMValidator now returns a list of Pydantic models (AlpacaItem)
        # We need to convert them back to dicts for the response_model
        validated_data_models = await slm_validator.validate_and_format_data(data)

        # Using .model_dump() to convert Pydantic models to dictionaries
        validated_data_dicts = [
            item.model_dump() for item in validated_data_models
        ]

        return validated_data_dicts

    except DataValidationError as e:
        logger.error(f"Data validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("An unexpected error occurred during SLM data validation.")
        raise HTTPException(status_code=500, detail="An internal error occurred.")