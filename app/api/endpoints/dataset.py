from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

from app.services.dataset_analysis import dataset_analysis_service

router = APIRouter()

class DatasetConfig(BaseModel):
    id: str = Field(description="Hugging Face Dataset ID")
    config: Optional[str] = Field("default", description="Dataset config name")
    split: Optional[str] = Field("train", description="Split to analyze")

class AdvancedDatasetAnalysisRequest(BaseModel):
    datasets: List[DatasetConfig] = Field(description="List of datasets to interleave and analyze")
    target_model: str = Field(default="meta-llama/Meta-Llama-3-8B", description="Target SLM for format & tokenizer compatibility checks")

@router.post("/analyze", response_model=Dict[str, Any])
async def analyze_dataset_endpoint(request: AdvancedDatasetAnalysisRequest):
    """
    Analyzes one or multiple Hugging Face datasets for SLM fine-tuning suitability.
    """
    if not request.datasets:
        raise HTTPException(status_code=400, detail="Must provide at least one dataset configuration.")
        
    dataset_configs_dicts = [d.dict() for d in request.datasets]
    
    result = await dataset_analysis_service.analyze_datasets(dataset_configs_dicts, request.target_model)
    
    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
