"""Account API endpoints."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_db_engine, get_demo_mode
from ..schemas import AccountDetail, PaginatedAccounts
from ..services import account_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/accounts", tags=["Accounts"])


@router.get("/at-risk", response_model=PaginatedAccounts)
async def get_at_risk_accounts(
    risk_tier: str | None = None,
    plan_type: str | None = None,
    sort_by: str = "churn_probability",
    page: int = 1,
    per_page: int = 20,
    demo_mode: bool = Depends(get_demo_mode),
) -> PaginatedAccounts:
    """Get paginated at-risk accounts with filters."""
    engine = None if demo_mode else get_db_engine()
    data = account_service.get_at_risk_accounts(
        engine,
        demo_mode=demo_mode,
        risk_tier=risk_tier,
        plan_type=plan_type,
        sort_by=sort_by,
        page=page,
        per_page=per_page,
    )
    return PaginatedAccounts(**data)


@router.get("/{account_id}", response_model=AccountDetail)
async def get_account_detail(
    account_id: str,
    demo_mode: bool = Depends(get_demo_mode),
) -> AccountDetail:
    """Get detailed account information."""
    engine = None if demo_mode else get_db_engine()
    data = account_service.get_account_detail(engine, account_id, demo_mode)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
    return AccountDetail(**data)


@router.get("/{account_id}/shap")
async def get_account_shap(
    account_id: str,
    demo_mode: bool = Depends(get_demo_mode),
) -> dict[str, float]:
    """Get SHAP explanation for an account."""
    engine = None if demo_mode else get_db_engine()
    data = account_service.get_shap_explanation(engine, account_id, demo_mode)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found")
    return data
