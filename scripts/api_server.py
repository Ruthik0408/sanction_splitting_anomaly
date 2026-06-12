"""FastAPI service for duplicate purchase validation."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from scripts.config import database_settings
from scripts.duplicate_detector import DuplicatePurchaseDetector


class PurchaseCheckRequest(BaseModel):
    product_name: str = Field(..., min_length=1)
    fk_central_unit: int
    order_id: str = Field(..., min_length=1)
    supply_order_date: str = Field(..., min_length=1)


class BillCheckRequest(BaseModel):
    bill_id: int | str
    fk_central_unit: int
    order_id: str = Field(..., min_length=1)
    supply_order_date: str = Field(..., min_length=1)
    products: list[str] = Field(default_factory=list)


class AppState:
    detector: DuplicatePurchaseDetector | None = None
    startup_error: str | None = None


state = AppState()


@asynccontextmanager
async def lifespan(_: FastAPI):
    db = database_settings()
    try:
        state.detector = DuplicatePurchaseDetector(
            db_host=db.host,
            db_port=db.port,
            db_name=db.name,
            db_user=db.user,
            db_pass=db.password,
        )
        state.startup_error = None
    except Exception as exc:
        state.detector = None
        state.startup_error = str(exc)
    try:
        yield
    finally:
        if state.detector is not None:
            state.detector.close()


app = FastAPI(
    title="Duplicate Purchase Detector",
    version="1.0.0",
    lifespan=lifespan,
)


def detector() -> DuplicatePurchaseDetector:
    if state.detector is None:
        detail = state.startup_error or "Detector is not initialized"
        raise HTTPException(status_code=503, detail=detail)
    return state.detector


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    if state.detector is None:
        raise HTTPException(
            status_code=503,
            detail=state.startup_error or "Detector is not initialized",
        )
    try:
        state.detector.readiness_check()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "ready"}


@app.post("/check_purchase")
def check_purchase(payload: PurchaseCheckRequest) -> dict[str, Any]:
    try:
        return detector().check_purchase(
            product_name=payload.product_name,
            fk_central_unit=payload.fk_central_unit,
            order_id=payload.order_id,
            supply_order_date=payload.supply_order_date,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/check_bill")
def check_bill(payload: BillCheckRequest) -> dict[str, Any]:
    try:
        product_flags = [
            {
                "product_name": product_name,
                **detector().check_purchase(
                    product_name=product_name,
                    fk_central_unit=payload.fk_central_unit,
                    order_id=payload.order_id,
                    supply_order_date=payload.supply_order_date,
                ),
            }
            for product_name in payload.products
        ]
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "bill_id": payload.bill_id,
        "bill_flagged": any(flag["flagged"] for flag in product_flags),
        "products": product_flags,
    }
