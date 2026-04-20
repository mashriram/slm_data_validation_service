import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.api.router import api_router
from app.core.config import get_settings
from app.utils.http_client import close_http_client

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown events for the application."""
    yield
    # Gracefully close the httpx client on shutdown
    logging.info("Closing HTTP client session...")
    await close_http_client()
    logging.info("Application shutdown complete.")


settings = get_settings()

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="A service to validate, orchestrate generation, and standardize data for SLM fine-tuning.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

# Mount Gradio App
from app.gradio_ui import create_gradio_interface
import gradio as gr

gradio_app = create_gradio_interface()
app = gr.mount_gradio_app(app, gradio_app, path="/ui")


@app.get("/", tags=["Health Check"])
def read_root():
    """Health check endpoint."""
    return {
        "status": "ok",
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }
