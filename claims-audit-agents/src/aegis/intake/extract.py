"""Vision intake: claim-form image → validated `Claim`.

A vision-language model (NVIDIA Nemotron Nano VL by default) transcribes the form into a schema, then
deterministic checks catch transcription errors before anything is adjudicated:
  * line charges must sum to the form's total (box 28)
  * NPI check digit, enrolled member and provider, known CPT/ICD codes
Any failed check sends the form to manual intake instead of the audit.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass, field
from datetime import date

from pydantic import BaseModel, Field, ValidationError

from aegis.intake.forms import read_ground_truth
from aegis.llm.base import LLM, Message
from aegis.models import Claim
from aegis.npi import is_valid_npi
from aegis.reference import Reference


class FormLine(BaseModel):
    cpt: str
    modifiers: list[str] = Field(default_factory=list)
    charge: float
    units: int


class ExtractedForm(BaseModel):
    claim_id: str
    member_id: str
    provider_npi: str
    prior_auth_number: str | None = None
    referral_number: str | None = None
    submitted_date: date
    service_date: date
    place_of_service: str
    diagnosis_codes: list[str]
    lines: list[FormLine]
    total_charge: float


PROMPT = (
    "Transcribe this health insurance claim form exactly. Copy identifiers character by character. "
    "Use null for empty boxes. Dates as YYYY-MM-DD. Diagnosis codes in box 21 order."
)


@dataclass
class IntakeResult:
    claim: Claim | None
    issues: list[str] = field(default_factory=list)
    extracted: ExtractedForm | None = None

    @property
    def ok(self) -> bool:
        return self.claim is not None and not self.issues


class VisionExtractor:
    def __init__(self, llm: LLM, reference: Reference) -> None:
        self.llm = llm
        self.reference = reference

    async def extract(self, png: bytes, clinical_note: str | None = None) -> IntakeResult:
        data_uri = "data:image/png;base64," + base64.b64encode(png).decode()
        messages: list[Message] = [
            {
                "role": "user",
                "content": [{"type": "text", "text": PROMPT}, {"type": "image_url", "image_url": {"url": data_uri}}],
            }
        ]
        try:
            form = await self.llm.complete_json(messages, ExtractedForm)
        except Exception as exc:
            return IntakeResult(None, [f"extraction failed: {exc}"[:300]])
        return self.validate(form, clinical_note)

    def validate(self, form: ExtractedForm, clinical_note: str | None) -> IntakeResult:
        issues = []
        line_total = round(sum(line.charge for line in form.lines), 2)
        if abs(line_total - form.total_charge) > 0.01:
            issues.append(f"line charges sum to {line_total:,.2f} but total charge is {form.total_charge:,.2f}")
        if not is_valid_npi(form.provider_npi):
            issues.append(f"NPI {form.provider_npi} fails check digit (possible transcription error)")
        if form.member_id not in self.reference.members:
            issues.append(f"member {form.member_id} not found")
        unknown_cpt = [ln.cpt for ln in form.lines if ln.cpt not in self.reference.cpt]
        unknown_dx = [c for c in form.diagnosis_codes if c not in self.reference.icd]
        if unknown_cpt:
            issues.append(f"unknown CPT code(s): {', '.join(unknown_cpt)}")
        if unknown_dx:
            issues.append(f"unknown ICD-10 code(s): {', '.join(unknown_dx)}")
        try:
            claim = Claim(
                claim_id=form.claim_id,
                member_id=form.member_id,
                provider_npi=form.provider_npi,
                service_date=form.service_date,
                submitted_date=form.submitted_date,
                place_of_service=form.place_of_service,
                diagnosis_codes=form.diagnosis_codes,
                lines=[{"line_no": i + 1, **ln.model_dump()} for i, ln in enumerate(form.lines)],  # type: ignore[misc]
                prior_auth_number=form.prior_auth_number or None,
                referral_number=form.referral_number or None,
                clinical_note=clinical_note,
            )
        except ValidationError as exc:
            return IntakeResult(None, [*issues, f"invalid claim: {exc.errors()[0]['msg']}"], form)
        return IntakeResult(claim, issues, form)


class OfflineVisionLLM:
    """Test double: returns the ground truth embedded in forms rendered by `render_claim_form`."""

    name = "offline-form-reader"

    async def complete_json(self, messages: list[Message], schema: type) -> ExtractedForm:
        content = messages[-1]["content"]
        assert isinstance(content, list)
        url = next(part["image_url"]["url"] for part in content if part.get("type") == "image_url")
        truth = read_ground_truth(base64.b64decode(url.split(",", 1)[1]))
        if truth is None:
            raise ValueError("offline reader only supports forms rendered by Aegis")
        claim = Claim.model_validate(truth)
        return ExtractedForm(
            **claim.model_dump(
                include={
                    "claim_id",
                    "member_id",
                    "provider_npi",
                    "prior_auth_number",
                    "referral_number",
                    "submitted_date",
                    "service_date",
                    "place_of_service",
                    "diagnosis_codes",
                }
            ),
            lines=[FormLine(**ln.model_dump(include={"cpt", "modifiers", "charge", "units"})) for ln in claim.lines],
            total_charge=round(sum(ln.charge for ln in claim.lines), 2),
        )
