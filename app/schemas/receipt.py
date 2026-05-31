from pydantic import BaseModel


class ItemResult(BaseModel):
    name: str
    qty: int
    price: float


class ScanResult(BaseModel):
    success: bool
    store: str
    date: str
    items: list[ItemResult]
    total: float
