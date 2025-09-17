class DataValidationError(Exception):
    """Custom exception for data validation errors (e.g., missing columns)."""

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(self.detail)


class FileProcessingError(Exception):
    """Custom exception for errors during file parsing or reading."""

    def __init__(self, detail: str):
        self.detail = detail
        super().__init__(self.detail)


class DownstreamServiceError(Exception):
    """Custom exception for errors when communicating with the data generation service."""

    def __init__(self, service_name: str, status_code: int, detail: str):
        self.service_name = service_name
        self.status_code = status_code
        self.detail = f"Error from {service_name}: Status {status_code} - {detail}"
        super().__init__(self.detail)
