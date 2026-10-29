# Results summary

These compact tables transcribe the available manuscript. They are **reported results**, not a benchmark rerun or a claim that the current source/configuration reproduces every row. The paper PDF is intentionally not distributed in this release. Tables should be checked against the final published version when it becomes available.

## KITTI test set

AP R40 (%), IoU = 0.5 for Pedestrian and Cyclist. The two metrics and three difficulty levels are shown separately.

| Class | Metric | Easy | Moderate | Hard |
|---|---|---:|---:|---:|
| Pedestrian | AP3D | 17.32 | 11.22 | 9.56 |
| Pedestrian | APBEV | 18.91 | 12.50 | 11.77 |
| Cyclist | AP3D | 9.33 | 5.55 | 5.25 |
| Cyclist | APBEV | 10.44 | 6.95 | 6.77 |

These values come from manuscript Table III. Official test submission artifacts and a matching public detector checkpoint are not included. Model inference uses RGB and calibration; this does not mean training is free of LiDAR-derived supervision.

## Car validation ablation

Car AP3D R40 (%), IoU = 0.7; manuscript Table IV.

| Variant | DAE | OAE | Easy | Moderate | Hard |
|---|:---:|:---:|---:|---:|---:|
| Baseline | — | — | 25.50 | 17.79 | 15.26 |
| DAE only | ✓ | — | 28.60 | 19.76 | 17.19 |
| OAE only | — | ✓ | 29.40 | 21.34 | 18.85 |
| Combined | ✓ | ✓ | 29.29 | 21.17 | 19.17 |

An available epoch-80 OAE-only log matches 29.4009 / 21.3410 / 18.8510 before rounding. The other historical rows do not yet have a complete source/checkpoint/log mapping in this release. A separate recent combined-model run recorded **26.5944 / 19.4587 / 17.7380**; it must not be identified with the manuscript's combined row.

The historical variants involved source-level changes. The release YAML selects the current combined implementation and does not implement four independently reproducible ablation modes. The aggregate “Overall” column is omitted because its rounding/definition needs reconciliation with the original records. No statistically significant synergy or isolated occlusion robustness is inferred from these rows.

## Reported inference efficiency

Manuscript Table II; NVIDIA A100, inference batch size 1.

| Implementation | GFLOPs | Parameters (M) | FPS | GPU memory (GB) |
|---|---:|---:|---:|---:|
| CaDDN | 1970 | 67.5 | 4.2 | 14.3 |
| Proposed implementation | 913 | 48.6 | 9.1 | 8.7 |

These are complete-implementation measurements from the manuscript, not measurements made while preparing this release. Timing boundaries, warm-up, precision, CUDA synchronization and custom-op FLOP coverage are not fully captured in the available records. No throughput guarantee is made for the Docker recipe. The manuscript describes an A100 80 GB training host; the current validation machine has A100 40 GB devices.
