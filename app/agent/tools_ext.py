import os, statistics
from typing import List, Dict, Any
from .models import Offer

# simple static rates; extend via provider later
STATIC_RATES = {
    # base: RUB -> target
    ("RUB","USD"): 0.011,
    ("RUB","EUR"): 0.010,
    ("RUB","RUB"): 1.0,
    ("USD","RUB"): 90.0,
    ("EUR","RUB"): 100.0,
    ("USD","USD"): 1.0,
    ("EUR","EUR"): 1.0,
}

def compute_stats(prices: List[int]) -> Dict[str, float]:
    if not prices:
        return {"min": None, "max": None, "avg": None, "median": None}
    return {
        "min": int(min(prices)),
        "max": int(max(prices)),
        "avg": int(sum(prices)/len(prices)),
        "median": int(statistics.median(prices))
    }

def filter_offers(offers: List[dict], min_price: int = None, max_price: int = None, text_contains: str = "") -> List[dict]:
    out = []
    text_contains = (text_contains or "").lower()
    for o in offers:
        p = int(o.get("price", 0))
        title = str(o.get("title","")).lower()
        if min_price is not None and p < min_price:
            continue
        if max_price is not None and p > max_price:
            continue
        if text_contains and text_contains not in title:
            continue
        out.append(o)
    return out

def convert_currency(amount: int, src: str, dst: str) -> float:
    src = src.upper().replace("₽","RUB").replace("€","EUR").replace("$","USD")
    dst = dst.upper().replace("₽","RUB").replace("€","EUR").replace("$","USD")
    rate = STATIC_RATES.get((src, dst))
    if rate is None:
        raise ValueError(f"No rate for {src}->{dst}")
    return round(amount * rate, 2)

def normalize_offers_currency(offers: List[dict], target_currency: str = "RUB") -> List[dict]:
    out = []
    for o in offers:
        p = int(o["price"])
        cur = o["currency"]
        new_price = convert_currency(p, cur, target_currency)
        o2 = dict(o)
        o2["price"] = int(new_price) if target_currency == "RUB" else new_price
        o2["currency"] = target_currency
        out.append(o2)
    return out
