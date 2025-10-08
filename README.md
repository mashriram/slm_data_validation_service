# SLM Data Validation Service

This service provides an API for validating, orchestrating data generation, and standardizing data for SLM (Small Language Model) fine-tuning.

## Features

- **File Upload:** Process structured (`.csv`, `.json`) and unstructured (`.pdf`, `.docx`) files.
- **Data Generation:** For unstructured files, it communicates with a downstream data generation service to produce question/answer pairs.
- **Data Validation:** Ensures that the data conforms to the required format (e.g., Alpaca format).
- **Asynchronous Processing:** All file processing and validation are done asynchronously.

## Endpoints

### Health Check

- **GET /**: Returns the status of the service.

### V1 API

#### File-Based Validation and Processing

- **POST /api/v1/validate-file/**:
  - **Description:** Upload a file (`.csv`, `.json`, `.pdf`, `.docx`) to be processed and validated.
  - **Request Body:** `multipart/form-data` with a file.
  - **Response:** A JSON array of validated data in Alpaca format.

#### SLM Data Validation

- **POST /api/v1/validate-slm-data**:
  - **Description:** Validates a list of question/answer pairs from the SLM creation service. It checks for required fields, cleans the data, and returns it in the standard Alpaca format. Invalid rows are dropped.
  - **Request Body:** A JSON array of objects, where each object should have a `question` and `answer` key.
    ```json
    [
      {"question": "What is the capital of France?", "answer": "Paris"},
      {"question": "Invalid row", "answer": ""}
    ]
    ```
  - **Response:** A JSON array of validated data in Alpaca format.
    ```json
    [
      {
        "instruction": "What is the capital of France?",
        "input": "",
        "output": "Paris"
      }
    ]
    ```

## Running the Service Locally

1.  **Install Dependencies:**
    ```bash
    pip install -e .
    ```

2.  **Run the application:**
    ```bash
    uvicorn main:app --reload
    ```

## Running Tests

1.  **Install test dependencies:**
    ```bash
    pip install -e '.[test]'
    ```

2.  **Run tests:**
    ```bash
    pytest
    ```
