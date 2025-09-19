from typing import List
from llama_index.llms.openai import OpenAI
from llama_index.program.openai import OpenAIPydanticProgram
from .schemas_partsync import PartRecordPS

SPEC_PROMPT = """
You are an expert electronics datasheet parser.
Return ONLY valid JSON for the PartRecordPS schema. 
- Convert units to SI where appropriate; keep display strings where relevant.
- Add page-anchored provenance like: pdf://<file>#p=<n>:<anchor> for any extracted field.
- If unsure, set field to null and add a note to provenance.
- Do not hallucinate values not visible in the supplied text.
"""

_llm = OpenAI(model="gpt-4o-mini", temperature=0.1)

_program = OpenAIPydanticProgram.from_defaults(
    output_cls=PartRecordPS,
    llm=_llm,
    prompt=SPEC_PROMPT,
    max_retries=1,
)

def extract_records_from_docs(docs) -> List[dict]:
    out = []
    for d in docs:
        try:
            rec = _program(d.get_text())
            out.append(rec.model_dump())
        except Exception as e:
            print("extract error:", e)
    return out
