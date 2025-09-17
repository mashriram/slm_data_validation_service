from pydantic import BaseModel, Field


class AlpacaItem(BaseModel):
    """
    Defines the structure of a single item in the clean Alpaca-formatted dataset.
    This is the standardized output format of this service.
    """

    instruction: str = Field(
        ..., description="The instruction or question for the model."
    )
    input: str = Field(
        "", description="Optional context or input for the instruction. Can be empty."
    )
    output: str = Field(
        ..., description="The desired response or answer from the model."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "instruction": "What is the capital of France?",
                "input": "",
                "output": "The capital of France is Paris.",
            }
        }
