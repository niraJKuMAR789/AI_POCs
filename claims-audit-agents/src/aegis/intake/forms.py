"""Render a claim as a CMS-1500-style form image (synthetic test input for vision intake).

The ground-truth claim JSON is embedded in a PNG text chunk. Only the offline test double reads it;
the real path extracts from the pixels with a vision-language model.
"""

from __future__ import annotations

import io
import json

from PIL import Image, ImageDraw, ImageFont
from PIL.PngImagePlugin import PngInfo

from aegis.models import Claim

GROUND_TRUTH_KEY = "aegis-ground-truth"


def render_claim_form(claim: Claim) -> bytes:
    width, height = 1400, 900
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    big, font, small = (ImageFont.load_default(size=s) for s in (30, 20, 16))
    red = (180, 30, 30)

    draw.text((40, 25), "HEALTH INSURANCE CLAIM FORM (synthetic, CMS-1500 style)", fill=red, font=big)

    def box(x: int, y: int, w: int, h: int, label: str, value: str) -> None:
        draw.rectangle([x, y, x + w, y + h], outline=red, width=2)
        draw.text((x + 6, y + 4), label, fill=red, font=small)
        draw.text((x + 10, y + 26), value, fill="black", font=font)

    box(40, 80, 420, 60, "1a. INSURED'S ID NUMBER", claim.member_id)
    box(470, 80, 420, 60, "33a. BILLING PROVIDER NPI", claim.provider_npi)
    box(900, 80, 460, 60, "23. PRIOR AUTHORIZATION NUMBER", claim.prior_auth_number or "")
    box(40, 150, 420, 60, "17b. REFERRAL NUMBER", claim.referral_number or "")
    box(470, 150, 420, 60, "DATE SUBMITTED", claim.submitted_date.isoformat())
    box(900, 150, 460, 60, "CLAIM NUMBER", claim.claim_id)
    letters = "ABCDEFGHIJKL"
    dx = "   ".join(f"{letters[i]}. {code}" for i, code in enumerate(claim.diagnosis_codes))
    box(40, 220, 1320, 60, "21. DIAGNOSIS OR NATURE OF ILLNESS OR INJURY (ICD-10)", dx)

    headers = ["24A. DATE OF SERVICE", "B. POS", "D. CPT/HCPCS", "MODIFIER", "F. $ CHARGES", "G. UNITS"]
    xs = [40, 330, 470, 700, 900, 1160, 1360]
    y = 300
    for i, head in enumerate(headers):
        draw.rectangle([xs[i], y, xs[i + 1], y + 34], outline=red, width=2)
        draw.text((xs[i] + 6, y + 8), head, fill=red, font=small)
    for line in claim.lines:
        y += 46
        values = [
            claim.service_date.isoformat(),
            claim.place_of_service,
            line.cpt,
            " ".join(line.modifiers),
            f"{line.charge:,.2f}",
            str(line.units),
        ]
        for i, value in enumerate(values):
            draw.rectangle([xs[i], y, xs[i + 1], y + 40], outline=red, width=1)
            draw.text((xs[i] + 10, y + 10), value, fill="black", font=font)
    total = sum(line.charge for line in claim.lines)
    box(900, 800, 460, 60, "28. TOTAL CHARGE", f"{total:,.2f}")

    meta = PngInfo()
    meta.add_text(GROUND_TRUTH_KEY, claim.model_dump_json())
    out = io.BytesIO()
    img.save(out, format="PNG", pnginfo=meta)
    return out.getvalue()


def read_ground_truth(png: bytes) -> dict | None:
    with Image.open(io.BytesIO(png)) as img:
        raw = img.text.get(GROUND_TRUTH_KEY) if hasattr(img, "text") else None
    return json.loads(raw) if raw else None
