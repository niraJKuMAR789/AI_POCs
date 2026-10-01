# Aegis evaluation report

- **provider**: `offline`  **reviewer model**: `offline-keyword-baseline`
- **claims**: 144 (56 need clinical/coding judgment)
- **rules engine vs injected anomalies**: precision 1.000, recall 1.000
- **audit ledger hash chain**: verified

| metric | rules only | rules + agents |
|---|---|---|
| STP (no human) | 61.1% | 100.0% |
| claim accuracy (auto-decided) | 100.0% | 90.3% |
| line accuracy (auto-decided) | 100.0% | 93.5% |
| judgment accuracy (auto-decided) | - | 75.0% |
| judgment escalated to human | 100.0% | 0.0% |
| overpaid $ (should deny, paid) | $0 | $12,120 |
| underpaid $ (should pay, denied) | $0 | $2,200 |
| latency p50 | - | 12 ms |

_LLM-only baseline requires a model provider (skipped in offline mode)._
