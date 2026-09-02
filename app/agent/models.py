from pydantic import BaseModel, Field
from typing import List, Optional

class Offer(BaseModel):
    title: str
    price: int
    currency: str
    url: str

class ExtractionResult(BaseModel):
    offers: List[Offer] = Field(default_factory=list)

class MonitorRequest(BaseModel):
    url: str
    max_items: int = 50

class AskRequest(BaseModel):
    question: str
    urls: List[str] = Field(default_factory=list)  # можно задать одну или несколько страниц
    max_items: int = 50
