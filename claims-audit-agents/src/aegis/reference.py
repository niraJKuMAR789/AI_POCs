"""Reference data and registries the rules engine consults."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel


class CPTInfo(BaseModel):
    description: str
    ref_cost: float
    mue: int = 1
    category: str = "procedure"
    requires_auth: bool = False
    necessity: list[str] | None = None
    policy: str | None = None
    age_min: int | None = None
    age_max: int | None = None
    review_level: bool = False


class NCCIEdit(BaseModel):
    column1: str
    column2: str
    modifier_allowed: bool


class Member(BaseModel):
    member_id: str
    date_of_birth: date
    gender: str
    plan_type: str  # HMO | PPO
    coverage_start: date
    coverage_end: date | None = None

    def age_on(self, day: date) -> int:
        years = day.year - self.date_of_birth.year
        return years - ((day.month, day.day) < (self.date_of_birth.month, self.date_of_birth.day))

    def active_on(self, day: date) -> bool:
        return self.coverage_start <= day and (self.coverage_end is None or day <= self.coverage_end)


class Provider(BaseModel):
    npi: str
    name: str
    specialty: str
    network: str  # in_network | out_of_network


class Authorization(BaseModel):
    auth_number: str
    member_id: str
    cpt: str
    valid_from: date
    valid_to: date


class Referral(BaseModel):
    referral_number: str
    member_id: str
    provider_npi: str


class HistoryLine(BaseModel):
    member_id: str
    provider_npi: str
    service_date: date
    cpt: str
    claim_id: str


@dataclass
class Reference:
    cpt: dict[str, CPTInfo]
    icd: dict[str, str]
    ncci: list[NCCIEdit]
    emergency_pos: set[str]
    timely_filing_days: int
    outlier_multiple: float
    members: dict[str, Member] = field(default_factory=dict)
    providers: dict[str, Provider] = field(default_factory=dict)
    authorizations: list[Authorization] = field(default_factory=list)
    referrals: list[Referral] = field(default_factory=list)
    history: list[HistoryLine] = field(default_factory=list)

    @classmethod
    def load(cls, reference_yaml: Path, registry_json: Path | None = None) -> Reference:
        raw = yaml.safe_load(reference_yaml.read_text())
        ref = cls(
            cpt={code: CPTInfo.model_validate(v) for code, v in raw["cpt"].items()},
            icd={str(k): v for k, v in raw["icd"].items()},
            ncci=[NCCIEdit.model_validate(e) for e in raw["ncci"]],
            emergency_pos=set(raw["emergency_pos"]),
            timely_filing_days=raw["timely_filing_days"],
            outlier_multiple=raw["outlier_multiple"],
        )
        if registry_json and registry_json.exists():
            ref.load_registry(json.loads(registry_json.read_text()))
        return ref

    def load_registry(self, data: dict) -> None:
        self.members = {m["member_id"]: Member.model_validate(m) for m in data["members"]}
        self.providers = {p["npi"]: Provider.model_validate(p) for p in data["providers"]}
        self.authorizations = [Authorization.model_validate(a) for a in data["authorizations"]]
        self.referrals = [Referral.model_validate(r) for r in data["referrals"]]
        self.history = [HistoryLine.model_validate(h) for h in data["history"]]

    def find_auth(self, auth_number: str | None, member_id: str, cpt: str, day: date) -> Authorization | None:
        if not auth_number:
            return None
        for auth in self.authorizations:
            if (
                auth.auth_number == auth_number
                and auth.member_id == member_id
                and auth.cpt == cpt
                and auth.valid_from <= day <= auth.valid_to
            ):
                return auth
        return None

    def has_referral(self, referral_number: str | None, member_id: str, npi: str) -> bool:
        return bool(referral_number) and any(
            r.referral_number == referral_number and r.member_id == member_id and r.provider_npi == npi
            for r in self.referrals
        )

    def duplicates(self, member_id: str, npi: str, day: date, cpt: str, claim_id: str) -> list[HistoryLine]:
        return [
            h
            for h in self.history
            if h.member_id == member_id
            and h.provider_npi == npi
            and h.service_date == day
            and h.cpt == cpt
            and h.claim_id != claim_id
        ]
