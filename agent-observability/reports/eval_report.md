# Argus evaluation report

- **investigator**: deterministic RCA (rules over analysis tools)
- **workload**: 3 simulated agent apps, 12 h each run, 7 injected incident types (9 incident×project pairs per run)
- **seeds**: 1, 2, 3
- **wall time**: 312 s

## Detection and root cause by incident intensity

| intensity | runs | recall | precision | median delay | RCA category | RCA exact cause | evidence valid | stored PII leaks |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 100.0% | 100.0% | 6 min | 100.0% | 100.0% | 100.0% | 0 |
| 0.5 | 3 | 100.0% | 100.0% | 6 min | 100.0% | 100.0% | 100.0% | 0 |
| 0.25 | 3 | 85.2% | 100.0% | 7 min | 100.0% | 100.0% | 100.0% | 0 |
| 0.15 | 3 | 66.7% | 100.0% | 14 min | 94.4% | 94.4% | 94.4% | 0 |

## Recall by incident type

| incident type | intensity 1 | intensity 0.5 | intensity 0.25 | intensity 0.15 |
|---|---|---|---|---|
| provider_degradation | 100.0% | 100.0% | 100.0% | 100.0% |
| tool_failure | 100.0% | 100.0% | 100.0% | 33.3% |
| prompt_regression | 100.0% | 100.0% | 100.0% | 0.0% |
| agent_regression | 100.0% | 100.0% | 66.7% | 66.7% |
| model_change | 100.0% | 100.0% | 100.0% | 100.0% |
| data_leak | 100.0% | 100.0% | 100.0% | 66.7% |
| traffic_surge | 100.0% | 100.0% | 0.0% | 33.3% |

## Missed incidents

- 0.25/seed 1: I7:traffic_surge@support-bot
- 0.25/seed 2: I7:traffic_surge@support-bot
- 0.25/seed 3: I4:agent_regression@research-agent
- 0.25/seed 3: I7:traffic_surge@support-bot
- 0.15/seed 1: I3:prompt_regression@support-bot
- 0.15/seed 1: I6:data_leak@claims-agent
- 0.15/seed 1: I7:traffic_surge@support-bot
- 0.15/seed 2: I2:tool_failure@support-bot
- 0.15/seed 2: I3:prompt_regression@support-bot
- 0.15/seed 2: I7:traffic_surge@support-bot
- 0.15/seed 3: I2:tool_failure@support-bot
- 0.15/seed 3: I3:prompt_regression@support-bot
- 0.15/seed 3: I4:agent_regression@research-agent
