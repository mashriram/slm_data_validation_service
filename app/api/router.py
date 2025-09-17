# app/api/router.py
from fastapi import APIRouter
from app.api.endpoints import validation

api_router = APIRouter()
api_router.include_router(validation.router, prefix="/v1")
