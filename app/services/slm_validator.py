import logging
from typing import Any, Dict, List, Optional

from app.core.config import get_settings
from app.schemas.alpaca import AlpacaItem
from app.utils.exceptions import DataValidationError

logger = logging.getLogger(__name__)
settings = get_settings()


class SLMValidator:
    """
    Validates and formats data from the SLM creation service.
    """

    def _validate_row(self, row: Dict[str, Any], index: int) -> Optional[AlpacaItem]:
        """
        Validates a single row of data.

        Args:
            row: A dictionary representing a single row of data.
            index: The index of the row in the dataset.

        Returns:
            An AlpacaItem if the row is valid, otherwise None.
        """
        question = row.get("question")
        answer = row.get("answer")

        if not isinstance(question, str) or not question.strip():
            logger.warning(f"Row {index + 1}: Missing or invalid 'question'. Skipping.")
            return None

        if not isinstance(answer, str) or not answer.strip():
            logger.warning(f"Row {index + 1}: Missing or invalid 'answer'. Skipping.")
            return None

        return AlpacaItem(
            instruction=question.strip(),
            input="",  # Defaulting to empty for standard Alpaca
            output=answer.strip(),
        )

    async def validate_and_format_data(
        self, data: List[Dict[str, Any]]
    ) -> List[AlpacaItem]:
        """
        Validates and formats the data asynchronously.

        Args:
            data: A list of dictionaries representing the data from the SLM creation service.

        Returns:
            A list of validated and formatted AlpacaItem objects.
        """
        if not isinstance(data, list):
            raise DataValidationError("Invalid data format: expected a list of objects.")

        validated_data = []
        for i, row in enumerate(data):
            validated_row = self._validate_row(row, i)
            if validated_row:
                validated_data.append(validated_row)

        if not validated_data:
            raise DataValidationError(
                "Data processing resulted in an empty dataset after validation."
            )

        logger.info(
            f"Successfully validated and formatted {len(validated_data)} out of {len(data)} rows."
        )
        return validated_data