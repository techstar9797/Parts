from pydantic import BaseModel, Field
from typing import Optional, List, Literal, Dict, Union

class Range(BaseModel):
    min: Optional[float] = None
    max: Optional[float] = None
    unit: str

class PackageInfo(BaseModel):
    name: str                  # "SOIC-8", "QFN-24"
    pins: Optional[int] = None
    pitch_mm: Optional[float] = None
    body_l_mm: Optional[float] = None
    body_w_mm: Optional[float] = None

class Lifecycle(BaseModel):
    status: Literal["ACTIVE","NRND","EOL","OBSOLETE","PREVIEW"]
    source: Optional[str] = None

class PinoutPin(BaseModel):
    pin: int
    name: str
    function: Optional[str] = None

JSONScalar = Union[str, float, int, None]

class PartRecordPS(BaseModel):
    # Identity
    mpn: str
    manufacturer: str
    category: str              # OpAmp, LDO, Logic, MOSFET, Resistor, Capacitor, MCU, etc.
    mouser_url: Optional[str] = None
    datasheet_url: Optional[str] = None
    
    # Package / form
    package: PackageInfo
    
    # Fit (electrical)
    v_range: Range             # supply voltage operating range
    temp_range_c: Range
    power_dissipation_mw: Optional[float] = None
    io_level: Optional[str] = None  # CMOS/TTL, 1.8V/3.3V tolerant
    
    # Function (category-specific knobs)
    attrs: Dict[str, JSONScalar] = Field(default_factory=dict)
    
    # Passives
    value_si: Optional[float] = None
    value_display: Optional[str] = None
    tolerance_pct: Optional[float] = None
    
    # Compliance & lifecycle
    rohs: Optional[bool] = None
    lifecycle: Optional[Lifecycle] = None
    
    # Commerce (optional enrichment)
    stock: Optional[int] = None
    unit_price_usd: Optional[float] = None
    
    # Provenance & quality
    provenance: List[str] = Field(default_factory=list)  # pdf://<file>#p=<n>:<anchor>
    confidence: Optional[float] = None
    
    # Pinout information
    pinout: Optional[List[PinoutPin]] = None

class BOMLinePS(BaseModel):
    line_no: int
    qty: int
    refdes: Optional[str] = None
    mpn: Optional[str] = None
    manufacturer: Optional[str] = None
    value_display: Optional[str] = None
    package_hint: Optional[str] = None

class ReplacementCandidate(BaseModel):
    mpn: str
    manufacturer: str
    score: float
    form_score: float
    fit_score: float
    func_score: float
    reasons: List[str]
    provenance: List[str]
    part: Optional[PartRecordPS] = None

class FFFWeights(BaseModel):
    form: float = 0.5
    fit: float = 0.3
    func: float = 0.2

class HardGates(BaseModel):
    rohs_match: bool = True
    lifecycle_max: str = "NRND"  # reject EOL/OBSOLETE unless override
    min_confidence: float = 0.7
