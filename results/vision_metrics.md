# Vision-model escalation — results

Model `openai/gpt-4o-mini` via OpenRouter, temperature 0, 512 px, detail=low. Ground truth = my blind labels. 208 images sent (17 with detected faces withheld for privacy); 0 errors/unparseable.

| Policy | n | Said delete | Precision of delete | Recall of my deletes | Silent failures (delete→I kept) | Unsure rate |
|---|---:|---:|---:|---:|---:|---:|
| Rules (pipeline status) | 208 | 74 | 85% | 50% | 10 | 0% |
| Vision model alone | 208 | 96 | 85% | 66% | 12 | 2% |
| Hybrid: rules, vision vetoes screenshots & judges the rules' "keep" residual | 208 | 104 | 83% | 69% | 16 | 2% |
| Screenshots — rules | 65 | 65 | 89% | 100% | 6 | 0% |
| Screenshots — vision | 65 | 64 | 89% | 98% | 6 | 0% |
| Rules' "no problem found" residual — vision | 134 | 30 | 77% | 37% | 6 | 3% |

## Vision decision vs my label (all sent)

| Vision \ I said | delete | keep | unsure |
|---|---:|---:|---:|
| delete | 82 | 12 | 2 |
| unsure | 5 | 0 | 0 |
| keep | 38 | 67 | 2 |

## Cost and latency (measured)

- Tokens: 630,656 in / 5,836 out → **$0.0981** for 208 images = **$0.000472 per image**
- Latency per call: median 2456 ms, p90 3381 ms; wall time 131 s with 4 parallel calls
- Escalating only the rules' "keep" residual (50% of this library) for a 30,000-item library: ≈ 15,000 calls ≈ **$7.07**; sending everything ≈ $14.15

Privacy: images were my own, uploaded with my consent for this run only, downscaled, faces withheld; the product design keeps a local-only mode with no upload.
