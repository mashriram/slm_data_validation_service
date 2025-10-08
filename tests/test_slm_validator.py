import pytest
from app.services.slm_validator import SLMValidator
from app.utils.exceptions import DataValidationError
from app.schemas.alpaca import AlpacaItem

@pytest.mark.asyncio
async def test_validate_and_format_data_success():
    """
    Tests that a valid dataset is processed correctly.
    """
    validator = SLMValidator()
    input_data = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "Who wrote Hamlet?", "answer": "William Shakespeare"},
    ]

    expected_output = [
        AlpacaItem(instruction="What is the capital of France?", input="", output="Paris"),
        AlpacaItem(instruction="Who wrote Hamlet?", input="", output="William Shakespeare"),
    ]

    result = await validator.validate_and_format_data(input_data)
    assert result == expected_output

@pytest.mark.asyncio
async def test_validate_and_format_data_skips_invalid_rows():
    """
    Tests that rows with missing 'question' or 'answer' are skipped.
    """
    validator = SLMValidator()
    input_data = [
        {"question": "What is the capital of France?", "answer": "Paris"},
        {"question": "", "answer": "This should be skipped"},
        {"question": "Who wrote Hamlet?", "answer": ""},
        {"question": "What is 2+2?", "answer": "4"},
    ]

    expected_output = [
        AlpacaItem(instruction="What is the capital of France?", input="", output="Paris"),
        AlpacaItem(instruction="What is 2+2?", input="", output="4"),
    ]

    result = await validator.validate_and_format_data(input_data)
    assert result == expected_output

@pytest.mark.asyncio
async def test_validate_and_format_data_empty_input():
    """
    Tests that an empty dataset raises a DataValidationError.
    """
    validator = SLMValidator()
    with pytest.raises(DataValidationError, match="Data processing resulted in an empty dataset after validation."):
        await validator.validate_and_format_data([])

@pytest.mark.asyncio
async def test_validate_and_format_data_invalid_format():
    """
    Tests that invalid data format (not a list) raises a DataValidationError.
    """
    validator = SLMValidator()
    with pytest.raises(DataValidationError, match="Invalid data format: expected a list of objects."):
        await validator.validate_and_format_data({"not": "a list"})

@pytest.mark.asyncio
async def test_validate_and_format_data_empty_after_validation():
    """
    Tests that a dataset that becomes empty after validation raises a DataValidationError.
    """
    validator = SLMValidator()
    input_data = [
        {"question": "", "answer": "This should be skipped"},
        {"question": "Who wrote Hamlet?", "answer": ""},
    ]
    with pytest.raises(DataValidationError, match="Data processing resulted in an empty dataset after validation."):
        await validator.validate_and_format_data(input_data)