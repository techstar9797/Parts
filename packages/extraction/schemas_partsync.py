from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Union

class Range(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    unit: str

class PackageInfo(BaseModel):
    name: str
    pins: Optional[int] = None
    pitch_mm: Optional[float] = None
    body_l_mm: Optional[float] = None
    body_w_mm: Optional[float] = None

class Lifecycle(BaseModel):
    status: Literal["ACTIVE","NRND","EOL","OBSOLETE","PREVIEW"]
    source: Optional[str] = None

JSONScalar = Union[str, float, int, None]

class PartRecordPS(BaseModel):
    mpn: str
    manufacturer: str
    category: str
    mouser_url: Optional[str] = None
    datasheet_url: Optional[str] = None
    package: PackageInfo
    v_range: Range
    temp_range_c: Range
    power_dissipation_mw: Optional[float] = None
    io_level: Optional[str] = None
    attrs: Dict[str, JSONScalar] = Field(default_factory=dict)
    value_si: Optional[float] = None
    value_display: Optional[str] = None
    tolerance_pct: Optional[float] = None
    rohs: Optional[bool] = None
    lifecycle: Optional[Lifecycle] = None
    stock: Optional[int] = None
    unit_price_usd: Optional[float] = None
    provenance: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None
