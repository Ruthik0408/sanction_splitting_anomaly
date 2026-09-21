"""FastAPI interface for historical product-similarity search."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date
from typing import Any, Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from scripts.config import BASE_DIR, database_settings
from scripts.product_similarity_service import ProductSimilarityService


FRONTEND_DIR = BASE_DIR / "frontend"
FRONTEND_DIST_DIR = FRONTEND_DIR / "dist"
UI_STATIC_DIR = FRONTEND_DIST_DIR if FRONTEND_DIST_DIR.exists() else FRONTEND_DIR


class PurchaseCheckRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    product_name: str = Field(..., min_length=1)
    fk_central_unit: int
    order_id: str = Field(..., min_length=1)
    supply_order_date: date
    llm_provider: Literal["auto", "vllm", "openai"] = "auto"


class AppState:
    similarity_service: ProductSimilarityService | None = None
    startup_error: str | None = None


state = AppState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        db = database_settings()
        state.similarity_service = ProductSimilarityService(
            db_host=db.host,
            db_port=db.port,
            db_name=db.name,
            db_user=db.user,
            db_pass=db.password,
        )
        state.startup_error = None
    except Exception as exc:
        state.similarity_service = None
        state.startup_error = str(exc)
    try:
        yield
    finally:
        if state.similarity_service is not None:
            state.similarity_service.close()


app = FastAPI(
    title="Product Similarity Service",
    version="1.0.0",
    lifespan=lifespan,
)


def similarity_service() -> ProductSimilarityService:
    if state.similarity_service is None:
        detail = state.startup_error or "Similarity service is not initialized"
        raise HTTPException(status_code=503, detail=detail)
    return state.similarity_service


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/ui")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    if state.similarity_service is None:
        raise HTTPException(
            status_code=503,
            detail=state.startup_error or "Similarity service is not initialized",
        )
    try:
        state.similarity_service.readiness_check()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready"}


@app.get("/existing_purchases")
def existing_purchases(
    query: str = "",
    fk_central_unit: int | None = None,
    limit: int = Query(default=25, ge=1, le=100),
) -> dict[str, Any]:
    try:
        return {
            "items": similarity_service().search_existing_purchases(
                query=query,
                fk_central_unit=fk_central_unit,
                limit=limit,
            )
        }
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/check_purchase")
def check_purchase(payload: PurchaseCheckRequest) -> dict[str, Any]:
    try:
        return similarity_service().check_purchase(
            product_name=payload.product_name,
            fk_central_unit=payload.fk_central_unit,
            order_id=payload.order_id,
            supply_order_date=payload.supply_order_date.isoformat(),
            llm_provider=payload.llm_provider,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


app.mount("/ui", StaticFiles(directory=UI_STATIC_DIR, html=True), name="ui")
