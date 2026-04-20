# app/api/router.py
from fastapi import APIRouter
from app.api.endpoints import validation, dataset

api_router = APIRouter()
api_router.include_router(validation.router, prefix="/v1")
api_router.include_router(dataset.router, prefix="/v1/dataset", tags=["Dataset Analysis"])
