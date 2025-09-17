import logging
from typing import List

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.schemas.alpaca import AlpacaItem
from app.services.data_processor import DataProcessor
from app.utils.exceptions import (
    DataValidationError,
    DownstreamServiceError,
    FileProcessingError,
)

router = APIRouter()
logger = logging.getLogger(__name__)


# Dependency injection for the processor to make testing easier
def get_data_processor() -> DataProcessor:
    return DataProcessor()


@router.post(
    "/validate-and-prepare-data/",
    response_model=List[AlpacaItem],
    summary="Process and Validate Data for SLM Training",
    tags=["Data Processing"],
)
async def validate_and_prepare_data(
    file: UploadFile = File(
        ..., description="The data file (CSV, JSON, PDF, or DOCX)."
    ),
    processor: DataProcessor = Depends(get_data_processor),
):
    """
    This endpoint is the single entry point for all training data.

    - **For CSV/JSON**: Validates the presence of 'question' and 'answer' columns and cleans the data.
    - **For PDF/DOCX**: Forwards the file to the `SLM-data-generation-service` to create a Q&A dataset.
    - **Output**: Returns a JSON array of objects in the clean, standardized Alpaca format, ready for fine-tuning.
    """
    logger.info(f"Received file for validation and preparation: {file.filename}")
    try:
        alpaca_dataset = await processor.process_file(file)
        logger.info(
            f"Successfully processed '{file.filename}'. Returning {len(alpaca_dataset)} items."
        )
        return alpaca_dataset
    except (DataValidationError, FileProcessingError) as e:
        logger.error(
            f"Validation or processing error for '{file.filename}': {e.detail}"
        )
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=e.detail)
    except DownstreamServiceError as e:
        logger.error(f"Downstream service error for '{file.filename}': {e.detail}")
        # Use 502 Bad Gateway to indicate the problem is with an upstream service
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=e.detail)
    except Exception as e:
        logger.exception(
            f"An unexpected internal error occurred while processing '{file.filename}': {e}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An internal server error occurred.",
        )
