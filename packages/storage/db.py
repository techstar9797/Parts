import os, json
from typing import List, Dict, Any, Iterable, Optional
from sqlalchemy import create_engine, text

DB_URL = os.getenv("DATABASE_URL") or os.getenv("POSTGRES_URL") or     f"postgresql://{os.getenv('POSTGRES_USER','partsync')}:{os.getenv('POSTGRES_PASSWORD','partsync')}@{os.getenv('POSTGRES_HOST','localhost')}:{os.getenv('POSTGRES_PORT','5432')}/{os.getenv('POSTGRES_DB','partsync')}"

_engine = None
try:
    _engine = create_engine(DB_URL, pool_pre_ping=True)
    with _engine.begin() as cx:
        cx.execute(text("""
        CREATE TABLE IF NOT EXISTS parts(
            mpn TEXT PRIMARY KEY,
            manufacturer TEXT,
            category TEXT,
            data JSONB
        );
        """))
except Exception as e:
    print("DB init failed:", e)

def upsert_records(records: List[Dict[str, Any]]) -> int:
    if not _engine:
        return 0
    n = 0
    with _engine.begin() as cx:
        for r in records:
            mpn = r.get("mpn")
            if not mpn: continue
            cx.execute(text("INSERT INTO parts (mpn, manufacturer, category, data) VALUES (:mpn, :man, :cat, CAST(:data AS JSONB)) ON CONFLICT (mpn) DO UPDATE SET manufacturer=EXCLUDED.manufacturer, category=EXCLUDED.category, data=EXCLUDED.data"),
                       dict(mpn=mpn, man=r.get("manufacturer"), cat=r.get("category"), data=json.dumps(r)))
            n += 1
    return n

def search_parts(q: Optional[str]=None, category: Optional[str]=None, manufacturer: Optional[str]=None, limit:int=20, offset:int=0):
    if not _engine:
        return {"items": [], "total": 0}
    where = []
    params = {}
    if q:
        where.append("(mpn ILIKE :q OR manufacturer ILIKE :q)")
        params["q"] = f"%{q}%"
    if category:
        where.append("category = :cat")
        params["cat"] = category
    if manufacturer:
        where.append("manufacturer = :man")
        params["man"] = manufacturer
    wh = (" WHERE " + " AND ".join(where)) if where else ""
    with _engine.begin() as cx:
        total = cx.execute(text(f"SELECT COUNT(*) FROM parts{wh}"), params).scalar() or 0
        rows = cx.execute(text(f"SELECT mpn, data FROM parts{wh} ORDER BY mpn LIMIT :lim OFFSET :off"),
                          {**params, "lim": limit, "off": offset}).fetchall()
        items = [dict(mpnn=r[0], **(r[1])) if isinstance(r[1], dict) else dict(mpn=r[0], **json.loads(r[1])) for r in rows]
    return {"items": items, "total": total, "limit": limit, "offset": offset}

def get_part(mpn: str) -> Dict[str, Any]:
    if not _engine:
        return {}
    with _engine.begin() as cx:
        row = cx.execute(text("SELECT data FROM parts WHERE mpn=:mpn"), {"mpn": mpn}).fetchone()
        if not row: return {}
        return row[0] if isinstance(row[0], dict) else json.loads(row[0])

def iter_all_parts(exclude_mpn: Optional[str] = None) -> Iterable[Dict[str, Any]]:
    if not _engine:
        return []
    with _engine.begin() as cx:
        if exclude_mpn:
            rows = cx.execute(text("SELECT data FROM parts WHERE mpn != :mpn"), {"mpn": exclude_mpn}).fetchall()
        else:
            rows = cx.execute(text("SELECT data FROM parts")).fetchall()
    for r in rows:
        yield r[0] if isinstance(r[0], dict) else json.loads(r[0])
