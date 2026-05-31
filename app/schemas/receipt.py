from pydantic import BaseModel


class ItemResult(BaseModel):
    name: str
    qty: int
    price: float


class TotalsResult(BaseModel):
    grand_total: float
    subtotal: float
    discount: float
    tax: float
    cash: float
    change: float


class ScanResult(BaseModel):
    success: bool
    store: str
    date: str
    items: list[ItemResult]
    total: float
    totals: TotalsResult
    address: str
    raw_lines: list[dict]
    stats: dict
    processing_time_ms: int
