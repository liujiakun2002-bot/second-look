# Hybrid ranking — offline simulation

Gold set: 300 blind labels; first 50 scored. Rule scores replaced on 134 residual items by the vision decision (delete 0.75 / unsure 0.45 / keep 0.15). Simulated on the same labels; not a product run.

| Ordering | GB freed @50 | Precision @50 | Byte recall @50 |
|---|---:|---:|---:|
| Largest files first (no judgement) | 0.135 | 48% | 49% |
| Second Look: rules p(delete) x size | 0.095 | 26% | 35% |
| Hybrid (simulated): rules + vision on residual, x size | 0.126 | 48% | 46% |
| Hybrid (simulated), p(delete) only | 0.045 | 34% | 17% |
