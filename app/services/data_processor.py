import io
import json
import logging
from typing import Any, Dict, List

import httpx
import pandas as pd
from fastapi import UploadFile

from app.core.config import get_settings
from app.schemas.alpaca import AlpacaItem
from app.utils.exceptions import (
    DataValidationError,
    DownstreamServiceError,
    FileProcessingError,
)
from app.utils.http_client import get_http_client

logger = logging.getLogger(__name__)
settings = get_settings()


class DataProcessor:
    """Handles the core logic of validating, orchestrating generation, and transforming data."""

    def _validate_structured_data(self, data: List[Dict[str, Any]], filename: str):
        """Validates that data from CSV/JSON is not empty and has the required columns."""
        if not data:
            raise DataValidationError(
                f"The file '{filename}' is empty or contains no data."
            )

        first_item_keys = data[0].keys()
        if not all(col in first_item_keys for col in settings.EXPECTED_COLUMNS):
            raise DataValidationError(
                f"Invalid data structure in '{filename}'. "
                f"Required columns are: {settings.EXPECTED_COLUMNS}. "
                f"Found: {list(first_item_keys)}."
            )

    def _convert_to_alpaca(self, data: List[Dict[str, Any]]) -> List[AlpacaItem]:
        """Converts a list of dictionaries to a list of validated AlpacaItem objects."""
        alpaca_data = []
        for i, item in enumerate(data):
            question = item.get("question")
            answer = item.get("answer")

            # Basic data cleaning: skip rows with empty or non-string values
            if not (
                isinstance(question, str)
                and isinstance(answer, str)
                and question.strip()
                and answer.strip()
            ):
                logger.warning(f"Skipping row {i + 1} due to missing or invalid data.")
                continue

            alpaca_data.append(
                AlpacaItem(
                    instruction=question.strip(),
                    input="",  # Defaulting to empty input as per standard Alpaca format
                    output=answer.strip(),
                )
            )
        return alpaca_data

    async def _process_structured_file(self, file: UploadFile) -> List[Dict[str, Any]]:
        """Parses CSV or JSON files into a list of dictionaries."""
        try:
            content = await file.read()
            if file.filename.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(content))
                return df.to_dict(orient="records")
            elif file.filename.endswith(".json"):
                return json.loads(content)
        except Exception as e:
            raise FileProcessingError(
                f"Failed to read or parse file '{file.filename}': {e}"
            )

    async def _generate_data_from_unstructured_file(
        self, file: UploadFile
    ) -> List[Dict[str, Any]]:
        """Forwards a file to the data generation service and returns its response."""
        logger.info(f"Forwarding '{file.filename}' to data generation service.")
        async with get_http_client() as client:
            files = {"file": (file.filename, await file.read(), file.content_type)}
            try:
                response = await client.post(
                    settings.DATA_GENERATION_SERVICE_URL,
                    files=files,
                    timeout=settings.DATA_GENERATION_SERVICE_TIMEOUT,
                )
                response.raise_for_status()  # Raises HTTPStatusError for 4xx/5xx responses
                # The generation service returns a JSON with 'data' key
                return response.json().get("data", [])
            except httpx.TimeoutException:
                raise DownstreamServiceError(
                    "Data Generation Service", 504, "Request timed out."
                )
            except httpx.HTTPStatusError as e:
                # Forward the error from the downstream service
                detail = e.response.json().get("detail", e.response.text)
                raise DownstreamServiceError(
                    "Data Generation Service", e.response.status_code, detail
                )
            except Exception as e:
                raise DownstreamServiceError(
                    "Data Generation Service",
                    500,
                    f"An unexpected error occurred during communication: {e}",
                )

    async def process_file(self, file: UploadFile) -> List[AlpacaItem]:
        """Main orchestration method to process any uploaded file."""
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in settings.ALLOWED_EXTENSIONS:
            raise DataValidationError(
                f"File type '.{file_extension}' is not supported. Allowed types are: {settings.ALLOWED_EXTENSIONS}"
            )

        if file_extension in ["csv", "json"]:
            raw_data = await self._process_structured_file(file)
            self._validate_structured_data(raw_data, file.filename)
        elif file_extension in ["pdf", "docx"]:
            raw_data = await self._generate_data_from_unstructured_file(file)
            # We trust the downstream service's format, but a validation step here could be added for more robustness
        else:
            # This case is redundant due to the initial check but is good for safety.
            raise DataValidationError(
                "Internal error: unsupported file format passed initial check."
            )

        if not raw_data:
            raise FileProcessingError(
                "No data could be processed or generated from the file."
            )

        cleaned_alpaca_data = self._convert_to_alpaca(raw_data)

        if not cleaned_alpaca_data:
            raise FileProcessingError(
                "Data processing resulted in an empty dataset after cleaning. Check content for valid question/answer pairs."
            )

        return cleaned_alpaca_data
