# Benchmarks

Numbers below are from [`scripts/compare_benchmarks.py`](https://github.com/LakoreAI/kalbee/blob/main/scripts/compare_benchmarks.py),
run head-to-head against [FilterPy](https://github.com/rlabbe/filterpy),
[pykalman](https://github.com/pykalman/pykalman), and
[simdkalman](https://github.com/oseiskar/simdkalman) on the *same* task, with
the *same* `F`/`Q`/`H`/`R` and the *same* noisy sine-wave measurements. This
page is reproducible — run the script yourself:

```bash
uv sync --group benchmark   # or: pip install filterpy pykalman simdkalman
uv run python scripts/compare_benchmarks.py
```

## Correctness

All three linear-KF implementations produce **bit-for-bit identical position
RMSE** (0.6380) on the shared task — a solid cross-check that kalbee's
`KalmanFilter` is implementing the standard equations correctly, not just
"close enough."

## Single-series speed

| Library    | Time / run (ms) | Position RMSE |
|---|---:|---:|
| filterpy   |  6.9  | 0.6380 |
| **kalbee** | 12.5  | 0.6380 |
| simdkalman | 25.5  | 0.6745 |
| pykalman   | 34.1  | 0.6380 |

For a single bare `predict`/`update` loop, FilterPy's minimal, pedagogical
implementation has less per-call overhead than kalbee. kalbee is faster than
both pykalman and simdkalman's single-series path here — but if all you need
is one filter stepping through one series, FilterPy is hard to beat on raw
speed (it just does far less: no batching, no 17 other filter types, no
tracking/smoothing/learning modules).

## Vectorized: many independent series at once

This is where it matters in practice — tracking hundreds or thousands of
targets, or backtesting a filter over many time series. simdkalman's whole
pitch is speed here; kalbee has a purpose-built filter for exactly this,
[`VectorizedKalmanFilter`](filters/vectorized_kalman_filter.md):

| Approach | 1,000 series (ms) |
|---|---:|
| kalbee, naive per-series loop | 13,442 |
| simdkalman, batched | 728 |
| **kalbee, `VectorizedKalmanFilter`** | **184** |

kalbee's vectorized filter is **~4x faster than simdkalman** and **~73x
faster than looping** over the same 1,000 series — because it batches every
filter through the same `F @ x`/`F @ P @ F.T` NumPy call instead of iterating
in Python, the same idea as simdkalman but built into the standard
`BaseFilter` interface (same `predict`/`update`/`filter_sequence` API as
every other kalbee filter, so it's a drop-in swap, not a separate library).

## Feature breadth

|  | kalbee | FilterPy | pykalman | simdkalman | Stone Soup |
|---|---:|---:|---:|---:|---:|
| Filter implementations | **18** | ~10 | 2 | 1 | ~6 |
| Multi-object tracking (SORT/JPDA/PMBM) | ✅ | ❌ | ❌ | ❌ | ✅ (heavier framework) |
| Smoothers (RTS/EKF/UKF/fixed-lag) | ✅ | RTS only | RTS only | ❌ | ✅ |
| Parameter learning (EM, online EM, NIS auto-tune) | ✅ | ❌ | EM only | ❌ | Partial |
| Neural hybrid filter (KalmanNet) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Vectorized/batched filtering | ✅ | ❌ | ❌ | ✅ | ❌ |
| pandas / Polars DataFrame integration | ✅ | ❌ | ❌ | ❌ | ❌ |
| scikit-learn `fit`/`transform` API | ✅ | ❌ | ❌ | ❌ | ❌ |
| Actively maintained (2026) | ✅ | Mostly dormant | Mostly dormant | Mostly dormant | ✅ (defence-oriented) |

Stone Soup is the closest thing to a "does more than kalbee" comparison, but
it's a full defence-grade tracking *framework* — heavier to learn, optimized
for algorithm research rather than drop-in speed. If you want the breadth of
a research framework with the ergonomics of a normal Python library, that's
the gap kalbee is aimed at.
