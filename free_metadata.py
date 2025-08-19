# free_metadata.py
import re
from typing import Dict, Any
import dateparser
from word2number import w2n

# simple amenity keywords; extend as you wish
AMENITY_MAP = {
  "pool": ["pool", "swimming", "swim"],
  "spa": ["spa", "massage"],
  "restaurant": ["restaurant", "dinner", "lunch", "breakfast", "buffet"],
  "wifi": ["wifi", "wi-fi", "internet"],
  "parking": ["parking", "park"],
  "gym": ["gym", "fitness"],
  "room_service": ["room service", "in-room dining", "order to room"],
  "housekeeping": ["housekeeping", "cleaning", "clean", "towel", "towels"],
  "front_desk": ["reception", "front desk"],
  "bar": ["bar", "cocktail"],
  "tour": ["tour", "trip", "guide", "excursion"]
}

NUM_RE = re.compile(r"\b\d{1,3}\b")
WORDS = {
  "one":1,"two":2,"three":3,"four":4,"five":5,"six":6,"seven":7,"eight":8,"nine":9,"ten":10
}

def _detect_amenity(text: str) -> str:
  t = text.lower()
  best = ("other", 0)
  for k, syns in AMENITY_MAP.items():
    score = sum(1 for s in syns if s in t)
    if score > best[1]: best = (k, score)
  return best[0] if best[1]>0 else "other"

def _detect_qty(text: str) -> int | None:
  t = text.lower()
  m = NUM_RE.search(t)
  if m: return int(m.group())
  for w,n in WORDS.items():
    if re.search(rf"\b{w}\b", t): return n
  try:
    return w2n.word_to_num(t)  # will try to parse “two towels”
  except Exception:
    return None

def _detect_when(text: str) -> str | None:
  dt = dateparser.parse(text, settings={"PREFER_DATES_FROM": "future"})
  return dt.isoformat() if dt else None

def tag_text_free(text: str, action_from_intent: str | None = None) -> Dict[str, Any]:
  amenity = _detect_amenity(text)
  qty = _detect_qty(text)
  when = _detect_when(text)
  action = action_from_intent or "other"
  return {"action": action, "amenity": amenity, "qty": qty, "when": when}
