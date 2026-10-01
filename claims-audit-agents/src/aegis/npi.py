"""NPI check digit: Luhn over the 9-digit base prefixed with the 80840 health-industry constant."""

from __future__ import annotations


def _luhn_sum(digits: str) -> int:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 0:  # positions that are doubled when the check digit is appended
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total


def npi_check_digit(base9: str) -> int:
    return (10 - (_luhn_sum("80840" + base9) % 10)) % 10


def is_valid_npi(npi: str) -> bool:
    return len(npi) == 10 and npi.isdigit() and npi[0] in "12" and npi_check_digit(npi[:9]) == int(npi[9])


def make_npi(base9: str) -> str:
    return base9 + str(npi_check_digit(base9))
