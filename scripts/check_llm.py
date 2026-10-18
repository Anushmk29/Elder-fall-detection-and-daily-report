"""Check the OpenRouter key/model: one text call, and one image call if a strip exists.

Usage: OPENROUTER_API_KEY=... python scripts/check_llm.py [strip.jpg]
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
from app import llm, pipeline  # noqa: E402

ai = llm.LLM(pipeline.DATA_DIR / "llm")
print(f"model={ai.model} key={'set' if ai.key else 'MISSING'} spent today=${ai.spent_today():.4f} cap=${ai.cap}")
print("text :", ai.chat([{"role": "user", "content": "Reply with exactly: OK"}], max_tokens=5))
strip = Path(sys.argv[1]) if len(sys.argv) > 1 else next((BASE / "examples").glob("*/strip1.jpg"), None)
if strip:
    print("image:", ai.caption(strip, "test"))
print(f"spent today=${ai.spent_today():.4f}")
