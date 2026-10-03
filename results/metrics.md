# Second Look — evaluation

Gold set: **300** blind labels from a fixed random sample of 300 (mix: {'delete': 141, 'keep': 152, 'unsure': 7}). Each ordering replayed over the gold set; first **50** scored.

| Ordering | GB freed @50 | Precision @50 | Byte recall @50 | Keeps shown as top-50 |
|---|---:|---:|---:|---:|
| Reverse-chronological (what shipped swipe apps do) | 0.023 | 36% | 9% | 30 |
| Largest files first (no judgement) | 0.135 | 48% | 49% | 25 |
| Rules only: hash duplicates + screenshots | 0.012 | 32% | 4% | 34 |
| Full pipeline, ranked by p(delete) only | 0.022 | 20% | 8% | 39 |
| Full pipeline, ranked by p(delete) x size  [Second Look] | 0.095 | 26% | 35% | 36 |
| Random order (expected value) | 0.045 | 47% | 17% | — |

## System recommendation vs my label

| System said \ I said | delete | keep | unsure |
|---|---:|---:|---:|
| delete | 71 | 56 | 3 |
| unsure | 3 | 7 | 1 |
| keep | 67 | 89 | 3 |

- Precision of the system's "delete" recommendations: **55%**
- Silent failures (system "delete", I "keep"): **56**
- Abstention rate (system "unsure"): **4%**; my labels on those: {'unsure': 1, 'keep': 7, 'delete': 3}

## By content type

| Type | n | I deleted | System said delete | Precision of its delete |
|---|---:|---:|---:|---:|
| burst | 33 | 6 | 19 | 21% |
| photo | 185 | 72 | 40 | 18% |
| screenshot | 71 | 60 | 71 | 85% |
| video | 11 | 3 | 0 | — |

Ranked (product) mode: 260 photos seen, 0.589 GB marked → **0.113 GB per 50 swipes** (only what the ranker surfaced; not comparable across orderings).
