from typing import Dict, Any, List

DEFAULT_WEIGHTS = {"form": 0.5, "fit": 0.3, "func": 0.2}

def _form_score(t, c) -> float:
    s = 0.0; n = 0
    if t.get("package", {}).get("name") and c.get("package", {}).get("name"):
        n += 1; s += 1.0 if t["package"]["name"] == c["package"]["name"] else 0.0
    if t.get("package", {}).get("pins") and c.get("package", {}).get("pins"):
        n += 1; s += 1.0 if t["package"]["pins"] == c["package"]["pins"] else 0.0
    return s / n if n else 0.0

def _overlap(r1, r2):
    if not r1 or not r2: return 0.0
    a1, a2 = r1.get("min"), r1.get("max")
    b1, b2 = r2.get("min"), r2.get("max")
    if a1 is None or a2 is None or b1 is None or b2 is None: return 0.0
    inter = max(0.0, min(a2, b2) - max(a1, b1))
    union = max(a2, b2) - min(a1, b1)
    return inter/union if union>0 else 0.0

def _fit_score(t, c) -> float:
    v = _overlap(t.get("v_range"), c.get("v_range"))
    temp = _overlap(t.get("temp_range_c"), c.get("temp_range_c"))
    # equal weight
    return (v + temp) / 2.0

def _func_score(t, c) -> float:
    # simple Jaccard on attrs keys as a placeholder
    kt = set(k for k,v in (t.get("attrs") or {}).items() if v is not None)
    kc = set(k for k,v in (c.get("attrs") or {}).items() if v is not None)
    if not kt or not kc: return 0.0
    return len(kt & kc) / len(kt | kc)

def find_replacements(target: Dict[str, Any], candidates: List[Dict[str, Any]], weights=None, constraints=None):
    w = {**DEFAULT_WEIGHTS, **(weights or {})}
    results = []
    for c in candidates:
        # gates
        if constraints:
            if constraints.get("rohs") is True and c.get("rohs") is False:
                continue
        f = _form_score(target, c)
        e = _fit_score(target, c)
        b = _func_score(target, c)
        score = w["form"]*f + w["fit"]*e + w["func"]*b
        reasons = [f"form={f:.2f}", f"fit={e:.2f}", f"func={b:.2f}"]
        results.append({"mpn": c.get("mpn"), "manufacturer": c.get("manufacturer"), "score": round(score,4), "reasons": reasons})
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:50]
