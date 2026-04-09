# Learned refresh window prefetch PoC

This project is a **proof of concept** for a learned, timing-based prefetcher that tries to reduce load latency by prefetching before **recurring slow-access windows**.

It uses:
- real user-space loads
- `clflush`-based eviction
- `rdtscp` timing
- software prefetch hints
- online bucket models that learn recurring slow-load timing

It does **not** observe real DRAM refresh commands directly from user space.

So the right claim is:

> This PoC learns recurring slow-access timing patterns on real hardware and uses them to issue selective prefetches.

Not:

> This PoC directly measures DRAM refresh commands.

---

## What the PoC is trying to do

The original goal is to explore whether a prefetcher can reduce stalls caused by short recurring periods where memory access becomes expensive.

The current implementation does this by:

1. measuring real load latencies
2. identifying recurring **slow-load events**
3. grouping accesses into hashed **buckets**
4. learning a rough future timing model per bucket
5. prefetching selected future accesses when a bucket is near its next predicted slow window

There is also a global fallback mode in the code, but the most interesting results so far have come from `bucket-only` mode.

---

## Current design

### Core mechanisms

The PoC contains:

- a benchmark loop that repeatedly touches a working set of cache lines
- explicit eviction using `clflush`
- latency measurement using `rdtscp`
- a learned bucket model per address bucket
- an optional global timing model fallback
- a prefetch stage driven by predicted windows

### Bucket model

Each bucket learns online from recurring slow-load timing.

The iterative version maintains per-bucket state such as:
- estimated period
- jitter
- confidence
- quality
- prediction count
- useful prediction count
- false alarms
- suggested prefetch degree

The idea is to make the learner adapt instead of only using a fixed learned period.

---

## What this PoC is **not**

This is not:
- a kernel-space memory controller tracer
- a direct DRAM refresh monitor
- a hardware prefetcher implementation in silicon
- proof that the observed recurring slow windows are literally DRAM refresh windows

User space generally cannot see the true memory controller refresh schedule directly.

So this project is best understood as a:

> **real-memory timing-based learned prefetch experiment**

that may be useful for refresh-inspired ideas, but does not prove refresh causality by itself.

---

## Build

Compile with GCC or Clang on x86_64 Linux.

### GCC

```bash
g++ -O3 -std=c++20 -march=native dram_refresh_prefetch.cpp -o dram_refresh_prefetch
```

### Clang

```bash
clang++ -O3 -std=c++20 -march=native dram_refresh_prefetch.cpp -o dram_refresh_prefetch
```

---

## Basic run

Example:

```bash
./dram_refresh_prefetch \
  --working-set 128 \
  --iterations 1500 \
  --prefetch-count 8 \
  --lead-ns 120000 \
  --refresh-guard-ns 90000 \
  --model-buckets 65536 \
  --min-samples 2 \
  --period-tolerance-pct 30 \
  --confidence-threshold-pct 55 \
  --training-pct 50 \
  --min-outlier-cycles 140 \
  --outlier-sigma-x100 220 \
  --min-dev-cycles 4 \
  --learn-min-slow-cycles 150 \
  --learn-sigma-x100 160 \
  --learn-min-dev-cycles 2 \
  --min-bad-outliers-per-window 4 \
  --slow-threshold-cycles 200 \
  --very-slow-cycles 350 \
  --extreme-slow-cycles 500 \
  --predictor-mode bucket-only
```

---

## Important configuration flags

### Workload and benchmark shape

#### `--working-set`
Number of cache lines touched per iteration.

Higher values increase difficulty and cache pressure.

#### `--iterations`
How many total benchmark iterations to run.

More iterations usually means more stable statistics.

#### `--eviction-stride`
Controls the eviction pattern before measured loads.

---

### Prefetch behavior

#### `--prefetch-count`
Maximum number of candidate lines to prefetch for a predicted window.

This is one of the most important knobs.

- lower values are usually safer
- higher values may improve recall but also add pollution

#### `--lead-ns`
How early to issue prefetches before the predicted slow window.

#### `--refresh-guard-ns`
Additional timing slack around the predicted window.

Even though the name contains “refresh,” this is really a **predicted slow-window guard**, not proof of literal refresh timing.

---

### Bucket learner

#### `--model-buckets`
Number of bucket models used to group accesses.

In experiments so far, **larger bucket counts often helped** because they reduced aliasing between unrelated address regions.

#### `--min-samples`
Minimum number of accepted timing samples needed before a bucket can become meaningfully active.

#### `--period-tolerance-pct`
How much timing deviation is tolerated when matching recurring intervals.

#### `--confidence-threshold-pct`
Minimum confidence before a bucket is allowed to predict aggressively.

---

### Training split

#### `--training-pct`
Percentage of iterations used as training before evaluation begins.

Example:
- `50` means first half of iterations train the model
- second half is used for evaluation/reporting

---

### Reporting outlier thresholds

These are used mainly for evaluation and “bad window” reporting.

#### `--min-outlier-cycles`
Minimum latency required before a load can count as an outlier.

#### `--outlier-sigma-x100`
Outlier threshold expressed relative to deviation, scaled by 100.

#### `--min-dev-cycles`
Minimum deviation floor used when classifying outliers.

#### `--min-bad-outliers-per-window`
Minimum number of outlier loads required before an iteration is called a bad window.

---

### Learning thresholds

These control how easily the model learns from slow events.

#### `--learn-min-slow-cycles`
Minimum latency for a load to qualify as a learning event.

#### `--learn-sigma-x100`
Learning threshold relative to deviation, scaled by 100.

#### `--learn-min-dev-cycles`
Minimum deviation floor for learning-event classification.

These are intentionally separate from the reporting thresholds so that:
- learning can be looser
- reporting can remain stricter

---

### Tail classification

#### `--very-slow-cycles`
Threshold used for “very slow” load counts.

#### `--extreme-slow-cycles`
Threshold used for “extreme slow” load counts.

---

### Predictor mode

#### `--predictor-mode`
Available modes:
- `bucket-only`
- `global-only`
- `hybrid`

`bucket-only` is the most useful mode when testing whether per-region learning really works.

#### `--global-confidence-threshold-pct`
Confidence threshold for global prediction mode.

#### `--global-cooldown-pct`
Cooldown for the global fallback so it does not fire continuously.

---

## How to interpret the output

The program prints two main sections:
- `Baseline`
- `RefreshAwarePrefetch`

Then a `Comparison` section.

### Core latency metrics

#### `avg_cycles`
Mean measured load latency.

#### `p50`, `p90`, `p95`, `p97`, `p99`, `p99_5`
Percentile latencies.

#### `max_cycles`
Maximum observed load latency.

### Slow-load counts

#### `slow_loads`
Number of loads above the slow threshold.

#### `very_slow_loads`
Number of loads above `--very-slow-cycles`.

#### `extreme_slow_loads`
Number of loads above `--extreme-slow-cycles`.

These are often more informative than one percentile alone.

### Prediction metrics

#### `predicted_windows`
How many evaluation iterations were treated as predicted slow windows.

#### `loads_in_predicted_windows`
How many measured loads happened inside predicted windows.

#### `prefetch_instructions`
Number of software prefetch hints issued.

#### `learned_buckets`
How many bucket models became active enough to be considered learned.

#### `avg_active_bucket_confidence`
Average confidence among active buckets.

#### `avg_active_bucket_quality`
Average quality among active buckets.

#### `avg_active_bucket_degree`
Average suggested prefetch degree among active buckets.

### Window metrics

#### `actual_bad_windows`
Number of evaluation iterations that were classified as bad windows.

#### `useful_predicted_windows`
Predicted windows that were also counted as useful under the current reporting logic.

#### `avoided_bad_windows`
Predicted windows where the prefetch side avoided a bad-window classification.

#### `predicted_and_avoided_bad_windows`
Useful subset of predicted windows that actually avoided badness.

#### `window_precision`
How often predicted windows turned out useful.

#### `window_recall`
How often actual bad windows were correctly covered by useful predictions.

---

## Important caution about the metrics

Whole-window precision/recall can under-credit the prefetcher.

Why:
- a bucket can prefetch useful lines
- those lines can get faster
- but the **whole iteration** can still remain “bad” because of unrelated slow loads

So for now, the most trustworthy success metrics are usually:
- average latency
- slow / very slow / extreme slow counts
- `p99` / `p99.5`
- histogram movement
- active bucket confidence / quality / degree

---

## Reading the histogram

The histogram buckets show how load latencies are distributed:
- `histogram_lt64`
- `histogram_64_127`
- `histogram_128_191`
- `histogram_192_255`
- `histogram_256_319`
- `histogram_320_383`
- `histogram_384_plus`

A good prefetch result often looks like:
- more loads in the lower bands
- fewer loads in `384_plus`
- lower `p99` / `p99.5`

---

## Best current interpretation of the project

At this stage, the strongest defensible statement is:

> The PoC shows that a learned, bucket-based, real-memory timing prefetcher can reduce average latency and high-latency tail behavior on the tested workload.

The weaker statement is:

> It might be capturing refresh-like recurring slow windows, but this user-space benchmark does not directly prove that those windows are literal DRAM refresh events.

That distinction matters.

---

## Observations from current experiments

A few patterns have shown up repeatedly:

1. **Bucket-only mode can work**.
2. **Larger bucket counts often help** because they reduce aliasing.
3. **Smaller working sets are easier to learn cleanly**.
4. **Iterative online bucket updates improved results noticeably**.
5. Whole-window precision/recall can stay weak even when latency metrics improve strongly.

---

## Limitations

- user space cannot directly observe true DRAM refresh commands
- timing is noisy and affected by OS scheduling, interrupts, frequency changes, and background load
- access bucketing is a heuristic, not a true DRAM bank decoder
- prediction quality depends strongly on workload locality and repeatability
- baseline and prefetch runs are still real machine runs, so some run-to-run noise is unavoidable

---

## Good next improvements

The strongest next upgrades would be:

1. per-prefetched-line usefulness accounting
2. per-bucket success/failure quality tracking
3. stream-aware bucket keys
4. multi-resolution bucket models
5. CPU affinity and noise reduction
6. more explicit comparison modes between static and iterative learners

---

## Recommended starting points

### Performance-oriented bucket-only run

```bash
./dram_refresh_prefetch \
  --working-set 128 \
  --iterations 1500 \
  --prefetch-count 8 \
  --lead-ns 120000 \
  --refresh-guard-ns 90000 \
  --model-buckets 65536 \
  --min-samples 2 \
  --period-tolerance-pct 30 \
  --confidence-threshold-pct 55 \
  --training-pct 50 \
  --min-outlier-cycles 140 \
  --outlier-sigma-x100 220 \
  --min-dev-cycles 4 \
  --learn-min-slow-cycles 150 \
  --learn-sigma-x100 160 \
  --learn-min-dev-cycles 2 \
  --min-bad-outliers-per-window 4 \
  --very-slow-cycles 350 \
  --extreme-slow-cycles 500 \
  --predictor-mode bucket-only
```

### Learning-friendly run

```bash
./dram_refresh_prefetch \
  --working-set 64 \
  --iterations 1500 \
  --prefetch-count 8 \
  --lead-ns 120000 \
  --refresh-guard-ns 90000 \
  --model-buckets 4096 \
  --min-samples 2 \
  --period-tolerance-pct 30 \
  --confidence-threshold-pct 55 \
  --training-pct 50 \
  --min-outlier-cycles 140 \
  --outlier-sigma-x100 220 \
  --min-dev-cycles 4 \
  --learn-min-slow-cycles 110 \
  --learn-sigma-x100 160 \
  --learn-min-dev-cycles 2 \
  --min-bad-outliers-per-window 4 \
  --very-slow-cycles 300 \
  --extreme-slow-cycles 360 \
  --predictor-mode bucket-only
```

---

## Example Ouput
```
Real-memory learned-window prefetch PoC
Configuration
  lines: 32768
  working_set: 128
  iterations: 1500
  prefetch_count: 8
  lead_ns: 120000
  refresh_guard_ns: 90000
  eviction_stride: 8
  slow_threshold_cycles: 200
  model_bucket_count: 65536
  min_samples: 2
  min_period_ns: 1000
  max_period_ns: 50000000
  period_tolerance_pct: 30
  confidence_threshold_pct: 55
  training_pct: 50
  min_outlier_cycles: 140
  outlier_sigma_x100: 220
  min_dev_cycles: 4
  learn_min_slow_cycles: 150
  learn_sigma_x100: 160
  learn_min_dev_cycles: 2
  min_bad_outliers_per_window: 4
  very_slow_cycles: 350
  extreme_slow_cycles: 500
  predictor_mode: bucket-only
  global_confidence_threshold_pct: 75
  global_cooldown_pct: 80

Baseline
  total_loads: 192000
  avg_cycles: 219.55
  p50_cycles: 111
  p90_cycles: 407
  p95_cycles: 444
  p97_cycles: 481
  p99_cycles: 1184
  p99_5_cycles: 1554
  max_cycles: 75073
  slow_loads: 60326
  slow_ratio: 0.3142
  very_slow_loads: 60255
  extreme_slow_loads: 5534
  prefetch_instructions: 0
  loads_in_predicted_windows: 0
  predicted_windows: 0
  actual_bad_windows: 604
  useful_predicted_windows: 0
  missed_bad_windows: 604
  avoided_bad_windows: 0
  predicted_and_avoided_bad_windows: 0
  window_precision: 0.0000
  window_recall: 0.0000
  learned_buckets: 553
  avg_active_bucket_confidence: 0.6554
  avg_active_bucket_quality: 0.5000
  avg_active_bucket_degree: 4.00
  global_model_active: 1
  global_predicted_windows: 0
  learning_events: 37404
  trained_event_count: 37404
  severe_bad_windows: 274
  avg_outliers_per_eval_window: 11.53
  training_iterations: 750
  eval_iterations: 750
  flushes: 24000
  histogram_lt64: 0
  histogram_64_127: 122026
  histogram_128_191: 9648
  histogram_192_255: 17
  histogram_256_319: 30
  histogram_320_383: 8828
  histogram_384_plus: 51451

RefreshAwarePrefetch
  total_loads: 192000
  avg_cycles: 159.65
  p50_cycles: 111
  p90_cycles: 407
  p95_cycles: 407
  p97_cycles: 407
  p99_cycles: 518
  p99_5_cycles: 1036
  max_cycles: 9028
  slow_loads: 30506
  slow_ratio: 0.1589
  very_slow_loads: 30478
  extreme_slow_loads: 1929
  prefetch_instructions: 1716
  loads_in_predicted_windows: 1730
  predicted_windows: 406
  actual_bad_windows: 651
  useful_predicted_windows: 94
  missed_bad_windows: 339
  avoided_bad_windows: 94
  predicted_and_avoided_bad_windows: 94
  window_precision: 0.2315
  window_recall: 0.1444
  learned_buckets: 596
  avg_active_bucket_confidence: 0.6854
  avg_active_bucket_quality: 0.7129
  avg_active_bucket_degree: 6.64
  global_model_active: 1
  global_predicted_windows: 0
  learning_events: 37438
  trained_event_count: 37438
  severe_bad_windows: 355
  avg_outliers_per_eval_window: 10.86
  training_iterations: 750
  eval_iterations: 750
  flushes: 24000
  histogram_lt64: 0
  histogram_64_127: 150945
  histogram_128_191: 10549
  histogram_192_255: 10
  histogram_256_319: 13
  histogram_320_383: 7078
  histogram_384_plus: 23405

Comparison
  avg_cycle_reduction_pct: 27.29
  slow_load_reduction_pct: 49.43
  very_slow_reduction_pct: 49.42
  extreme_slow_reduction_pct: 65.14
  p95_delta_cycles: 37
  p99_delta_cycles: 666
  p99_5_delta_cycles: 518
```

An average cycle reduction of 27% and slow reduction up to 65%, wow...

---
## Final note

This project started as a refresh-inspired idea, but the current implementation is best understood as a **learned timing prefetcher PoC**.

That is still interesting and useful.

If later versions gain better locality keys, tighter usefulness accounting, and cleaner low-noise timing control, this can become a much stronger platform for evaluating refresh-aware or refresh-adjacent ideas.
