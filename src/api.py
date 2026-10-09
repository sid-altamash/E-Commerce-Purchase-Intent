"""FastAPI prediction service."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from src.config import configure_logging
from src.service import load_bundle, predict_session

configure_logging()
logger = logging.getLogger("purchase_intent.api")


class SessionInput(BaseModel):
    """Validated feature schema for a completed, late-session score."""

    model_config = ConfigDict(extra="forbid")

    Administrative: Annotated[int, Field(ge=0)]
    Administrative_Duration: Annotated[float, Field(ge=0)]
    Informational: Annotated[int, Field(ge=0)]
    Informational_Duration: Annotated[float, Field(ge=0)]
    ProductRelated: Annotated[int, Field(ge=0)]
    ProductRelated_Duration: Annotated[float, Field(ge=0)]
    BounceRates: Annotated[float, Field(ge=0, le=1)]
    ExitRates: Annotated[float, Field(ge=0, le=1)]
    PageValues: Annotated[float, Field(ge=0)]
    SpecialDay: Annotated[float, Field(ge=0, le=1)]
    Month: str
    OperatingSystems: str
    Browser: str
    Region: str
    TrafficType: str
    VisitorType: str
    Weekend: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _, metadata, _, _ = load_bundle()
    except FileNotFoundError:
        app.state.model_ready = False
        logger.warning("API started without a trained model artifact.")
    else:
        app.state.model_ready = True
        app.state.model_name = metadata["best_model"]
        logger.info("Loaded prediction artifact: %s", metadata["best_model"])
    yield


app = FastAPI(
    title="Purchase Intent API",
    description="Late-session e-commerce conversion scoring with explanations.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    response = await call_next(request)
    logger.info(
        "request method=%s path=%s status=%s",
        request.method,
        request.url.path,
        response.status_code,
    )
    return response


@app.get("/health")
def health(request: Request) -> JSONResponse:
    model_ready = bool(getattr(request.app.state, "model_ready", False))
    return JSONResponse(
        status_code=200 if model_ready else 503,
        content={
            "status": "ready" if model_ready else "not_ready",
            "model_ready": model_ready,
        },
    )


@app.post("/predict")
def predict(payload: SessionInput) -> dict:
    try:
        return predict_session(payload.model_dump())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except (ValueError, KeyError) as exc:
        logger.exception("Prediction failed validation.")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
