# app/api/router.py
from fastapi import APIRouter
from app.api.endpoints import validation, slm_validator

api_router = APIRouter()
api_router.include_router(validation.router, prefix="/v1")
api_router.include_router(slm_validator.router, prefix="/v1")
