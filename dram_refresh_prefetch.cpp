
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#if !defined(__x86_64__) && !defined(_M_X64)
#error "This PoC currently requires x86_64 for rdtscp/clflush/prefetch intrinsics."
#endif

namespace {

constexpr size_t kCacheLine = 64;

enum class PredictorMode { Hybrid, BucketOnly, GlobalOnly };

struct Config {
    size_t lines = 32768;
    size_t working_set = 128;
    size_t iterations = 1500;
    size_t prefetch_count = 12;
    size_t lead_ns = 120000;
    size_t refresh_guard_ns = 70000;
    size_t eviction_stride = 8;
    uint64_t slow_threshold_cycles = 180;
    uint32_t seed = 42;
    bool use_hugepage_hint = false;

    size_t model_bucket_count = 256;
    size_t min_samples = 3;
    size_t min_period_ns = 1000;
    size_t max_period_ns = 50000000;
    size_t period_tolerance_pct = 22;
    size_t confidence_threshold_pct = 68;
    size_t training_pct = 45;

    uint64_t min_outlier_cycles = 150;
    size_t outlier_sigma_x100 = 280;
    uint64_t min_dev_cycles = 7;
    uint64_t learn_min_slow_cycles = 110;
    size_t learn_sigma_x100 = 160;
    uint64_t learn_min_dev_cycles = 2;
    size_t min_bad_outliers_per_window = 4;
    uint64_t very_slow_cycles = 300;
    uint64_t extreme_slow_cycles = 360;
    PredictorMode predictor_mode = PredictorMode::Hybrid;
    size_t global_confidence_threshold_pct = 75;
    size_t global_cooldown_pct = 80;
};

struct Metrics {
    std::vector<uint32_t> latencies;
    uint64_t total_cycles = 0;
    uint64_t slow_loads = 0;
    uint64_t very_slow_loads = 0;
    uint64_t extreme_slow_loads = 0;
    uint64_t prefetched_loads = 0;
    uint64_t total_loads = 0;
    uint64_t flushes = 0;
    uint64_t prefetch_instructions = 0;
    uint64_t predicted_windows = 0;
    uint64_t global_predicted_windows = 0;
    uint64_t actual_bad_windows = 0;
    uint64_t useful_predicted_windows = 0;
    uint64_t missed_bad_windows = 0;
    uint64_t avoided_bad_windows = 0;
    uint64_t predicted_and_avoided_bad_windows = 0;
    uint64_t learned_buckets = 0;
    uint64_t learning_events = 0;
    uint64_t trained_event_count = 0;
    uint64_t training_iterations = 0;
    uint64_t eval_iterations = 0;
    uint64_t total_outliers_in_eval = 0;
    uint64_t severe_bad_windows = 0;
    bool global_model_active = false;
    double avg_active_bucket_confidence = 0.0;
    double avg_active_bucket_quality = 0.0;
    double avg_active_bucket_degree = 0.0;
};

struct WindowModel {
    uint64_t last_slow_ns = 0;
    uint64_t estimated_period_ns = 0;
    uint64_t predicted_next_ns = 0;
    uint64_t jitter_ns = 0;
    uint32_t samples = 0;
    uint32_t matches = 0;
    uint32_t mismatches = 0;
    uint32_t predictions = 0;
    uint32_t useful_predictions = 0;
    uint32_t false_alarms = 0;
    bool active = false;
    uint64_t last_fire_ns = 0;
    double confidence_value = 0.0;
    double quality_value = 0.0;
    size_t suggested_degree = 2;

    [[nodiscard]] double confidence() const { return confidence_value; }
    [[nodiscard]] double quality() const { return quality_value; }
};

struct LatencyStats {
    double ema = 0.0;
    double dev = 0.0;
    bool initialized = false;
};

inline uint64_t rdtscp() {
    unsigned aux = 0;
    return __rdtscp(&aux);
}

inline void serialize() {
    _mm_mfence();
    _mm_lfence();
}

inline uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

struct Buffer {
    std::vector<uint8_t> bytes;
    explicit Buffer(size_t line_count) : bytes(line_count * kCacheLine) {}
    uint8_t* line(size_t idx) { return bytes.data() + idx * kCacheLine; }
    const uint8_t* line(size_t idx) const { return bytes.data() + idx * kCacheLine; }
};

void touch_buffer(Buffer& buf) {
    volatile uint8_t sink = 0;
    for (size_t i = 0; i < buf.bytes.size(); i += 4096) {
        sink ^= buf.bytes[i];
        buf.bytes[i] = static_cast<uint8_t>(sink + i);
    }
}

std::vector<size_t> make_working_set(const Config& cfg) {
    std::vector<size_t> ws(cfg.working_set);
    std::mt19937 rng(cfg.seed);
    std::uniform_int_distribution<size_t> dist(0, cfg.lines - 1);
    size_t cursor = dist(rng);
    for (size_t i = 0; i < ws.size(); ++i) {
        if ((i % 8) == 0) cursor = dist(rng);
        cursor = (cursor + 1 + (rng() % 3)) % cfg.lines;
        ws[i] = cursor;
    }
    return ws;
}

inline uint32_t timed_load(const uint8_t* p) {
    serialize();
    const uint64_t t0 = rdtscp();
    volatile uint8_t v = *p;
    (void)v;
    serialize();
    const uint64_t t1 = rdtscp();
    return static_cast<uint32_t>(t1 - t0);
}

void evict_subset(Buffer& buf, const std::vector<size_t>& ws, const Config& cfg) {
    const size_t stride = std::max<size_t>(1, cfg.eviction_stride);
    for (size_t i = 0; i < ws.size(); i += stride) {
        _mm_clflush(buf.line(ws[i]));
    }
    serialize();
}

size_t bucket_for_line(size_t line_idx, const Config& cfg) {
    const size_t addr = line_idx * kCacheLine;
    const size_t x = (addr >> 6) ^ (addr >> 12) ^ (addr >> 18);
    return x & (cfg.model_bucket_count - 1);
}

void update_latency_stats(LatencyStats& s, double x) {
    if (!s.initialized) {
        s.ema = x;
        s.dev = x * 0.1;
        s.initialized = true;
        return;
    }
    const double alpha = 0.05;
    const double diff = x - s.ema;
    s.ema += alpha * diff;
    s.dev += alpha * (std::abs(diff) - s.dev);
}

bool is_outlier_thresh(const LatencyStats& s, uint32_t cyc, uint64_t min_cycles, size_t sigma_x100, uint64_t min_dev_cycles) {
    if (!s.initialized) return cyc >= min_cycles;
    const double sigma = static_cast<double>(sigma_x100) / 100.0;
    const double dev_floor = std::max<double>(min_dev_cycles, s.dev);
    const double dynamic_thr = s.ema + sigma * dev_floor;
    const double thr = std::max<double>(min_cycles, dynamic_thr);
    return static_cast<double>(cyc) >= thr;
}

bool is_outlier(const LatencyStats& s, uint32_t cyc, const Config& cfg) {
    return is_outlier_thresh(s, cyc, cfg.min_outlier_cycles, cfg.outlier_sigma_x100, cfg.min_dev_cycles);
}

bool is_learning_event(const LatencyStats& s, uint32_t cyc, const Config& cfg) {
    return is_outlier_thresh(s, cyc, cfg.learn_min_slow_cycles, cfg.learn_sigma_x100, cfg.learn_min_dev_cycles);
}

uint64_t tolerance_for(uint64_t period_ns, uint64_t jitter_ns, const Config& cfg) {
    const uint64_t pct = (period_ns * cfg.period_tolerance_pct) / 100ULL;
    const uint64_t adaptive = std::max<uint64_t>(pct, std::max<uint64_t>(2000, jitter_ns * 2));
    return std::min<uint64_t>(cfg.refresh_guard_ns, adaptive);
}

void recompute_model_state(WindowModel& model, const Config& cfg) {
    const double match_score = static_cast<double>(model.matches + 1) / static_cast<double>(model.matches + model.mismatches + 2);
    const double outcome_score = static_cast<double>(model.useful_predictions + 1) /
                                 static_cast<double>(model.predictions + 2);
    model.confidence_value = 0.65 * match_score + 0.35 * outcome_score;
    model.quality_value = outcome_score;
    if (model.quality_value > 0.55 && model.confidence_value > 0.60) {
        model.suggested_degree = 8;
    } else if (model.quality_value > 0.35 && model.confidence_value > 0.50) {
        model.suggested_degree = 4;
    } else {
        model.suggested_degree = 2;
    }
    model.active = model.samples >= cfg.min_samples &&
                   model.estimated_period_ns >= cfg.min_period_ns &&
                   model.confidence_value >= (static_cast<double>(cfg.confidence_threshold_pct) / 100.0);
}

void update_model(WindowModel& model, uint64_t now_ns_value, const Config& cfg) {
    if (model.last_slow_ns == 0) {
        model.last_slow_ns = now_ns_value;
        model.samples = 1;
        recompute_model_state(model, cfg);
        return;
    }
    const uint64_t dt = now_ns_value - model.last_slow_ns;
    model.last_slow_ns = now_ns_value;

    if (dt < cfg.min_period_ns || dt > cfg.max_period_ns) {
        ++model.mismatches;
        model.confidence_value *= 0.97;
        recompute_model_state(model, cfg);
        return;
    }
    if (model.estimated_period_ns == 0) {
        model.estimated_period_ns = dt;
        model.jitter_ns = std::max<uint64_t>(2000, dt / 8);
        model.predicted_next_ns = now_ns_value + dt;
        model.samples = std::max<uint32_t>(model.samples, 1U);
        ++model.matches;
        recompute_model_state(model, cfg);
        return;
    }

    const uint64_t tolerance = tolerance_for(model.estimated_period_ns, model.jitter_ns, cfg);
    const uint64_t diff = (dt > model.estimated_period_ns) ? (dt - model.estimated_period_ns)
                                                            : (model.estimated_period_ns - dt);
    if (diff <= tolerance) {
        model.estimated_period_ns = (model.estimated_period_ns * 7ULL + dt) / 8ULL;
        model.jitter_ns = (model.jitter_ns * 7ULL + diff) / 8ULL;
        ++model.matches;
        ++model.samples;
    } else {
        model.estimated_period_ns = (model.estimated_period_ns * 15ULL + dt) / 16ULL;
        model.jitter_ns = (model.jitter_ns * 7ULL + diff) / 8ULL;
        ++model.mismatches;
        if (model.mismatches > model.matches + 8) {
            model.confidence_value *= 0.9;
        }
    }

    model.predicted_next_ns = now_ns_value + model.estimated_period_ns;
    recompute_model_state(model, cfg);
}

bool near_predicted_window(const WindowModel& model, uint64_t now_ns_value, const Config& cfg) {
    if (!model.active || model.predicted_next_ns == 0 || model.estimated_period_ns == 0) return false;
    const uint64_t tolerance = tolerance_for(model.estimated_period_ns, model.jitter_ns, cfg);
    if (now_ns_value + cfg.lead_ns < model.predicted_next_ns) return false;
    if (now_ns_value > model.predicted_next_ns + tolerance) return false;
    return true;
}

bool global_near_predicted_window(const WindowModel& model, uint64_t now_ns_value, const Config& cfg) {
    if (!near_predicted_window(model, now_ns_value, cfg)) return false;
    if (model.confidence() < (static_cast<double>(cfg.global_confidence_threshold_pct) / 100.0)) return false;
    const uint64_t cooldown = (model.estimated_period_ns * cfg.global_cooldown_pct) / 100ULL;
    if (model.last_fire_ns && now_ns_value < model.last_fire_ns + std::max<uint64_t>(cooldown, 1000)) return false;
    return true;
}

void consume_prediction(WindowModel& model, uint64_t now_ns_value) {
    model.last_fire_ns = now_ns_value;
    ++model.predictions;
    if (model.estimated_period_ns == 0) return;
    if (model.predicted_next_ns <= now_ns_value) {
        const uint64_t period = std::max<uint64_t>(1, model.estimated_period_ns);
        while (model.predicted_next_ns <= now_ns_value) model.predicted_next_ns += period;
    }
}

void apply_prediction_feedback(WindowModel& model, bool useful, const Config& cfg) {
    if (model.predictions == 0) return;
    if (useful) {
        ++model.useful_predictions;
        model.confidence_value = std::min(1.0, model.confidence_value + 0.04);
    } else {
        ++model.false_alarms;
        model.confidence_value = std::max(0.0, model.confidence_value - 0.05);
    }
    recompute_model_state(model, cfg);
}

void apply_prediction_timing_feedback(WindowModel& model, uint64_t observed_ns, const Config& cfg) {
    if (model.last_fire_ns == 0 || model.estimated_period_ns == 0) return;
    const int64_t err = static_cast<int64_t>(observed_ns) - static_cast<int64_t>(model.predicted_next_ns);
    const uint64_t abs_err = static_cast<uint64_t>(std::llabs(err));
    model.jitter_ns = (model.jitter_ns * 7ULL + abs_err) / 8ULL;
    if (abs_err <= tolerance_for(model.estimated_period_ns, model.jitter_ns, cfg) * 2ULL) {
        const int64_t adjusted = static_cast<int64_t>(model.estimated_period_ns) + err / 8;
        model.estimated_period_ns = static_cast<uint64_t>(std::max<int64_t>(static_cast<int64_t>(cfg.min_period_ns), adjusted));
        model.predicted_next_ns = observed_ns + model.estimated_period_ns;
    }
    recompute_model_state(model, cfg);
}

void do_prefetch(Buffer& buf,
                 const std::vector<size_t>& ws,
                 const std::vector<uint8_t>& should_prefetch,
                 const Config& cfg,
                 Metrics& m) {
    std::vector<size_t> order(ws.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return should_prefetch[a] > should_prefetch[b];
    });
    size_t issued = 0;
    for (size_t idx : order) {
        if (issued >= cfg.prefetch_count) break;
        if (!should_prefetch[idx]) break;
        _mm_prefetch(reinterpret_cast<const char*>(buf.line(ws[idx])), _MM_HINT_T0);
        ++m.prefetch_instructions;
        ++issued;
    }
    _mm_pause(); _mm_pause(); _mm_pause();
}

Metrics run_case(Buffer& buf, const Config& cfg, bool refresh_aware_prefetch) {
    Metrics m;
    m.latencies.reserve(cfg.iterations * cfg.working_set);
    std::vector<size_t> ws = make_working_set(cfg);
    std::vector<WindowModel> models(cfg.model_bucket_count);
    std::vector<LatencyStats> latency_stats(cfg.model_bucket_count);
    WindowModel global_model;
    LatencyStats global_latency_stats;
    std::mt19937 rng(cfg.seed);
    const size_t training_iterations = (cfg.iterations * cfg.training_pct) / 100;
    m.training_iterations = training_iterations;
    m.eval_iterations = cfg.iterations - training_iterations;

    for (size_t it = 0; it < cfg.iterations; ++it) {
        const bool training_phase = it < training_iterations;
        if ((it % 8) == 0) {
            const size_t delta = 13 + (rng() % 19);
            for (size_t& idx : ws) idx = (idx + delta) % cfg.lines;
        }

        evict_subset(buf, ws, cfg);
        m.flushes += (ws.size() + cfg.eviction_stride - 1) / std::max<size_t>(1, cfg.eviction_stride);

        std::vector<uint8_t> should_prefetch(ws.size(), 0);
        bool predicted_window = false;
        bool used_global_prediction = false;
        std::vector<size_t> fired_buckets;
        if (refresh_aware_prefetch && !training_phase) {
            const bool bucket_mode = cfg.predictor_mode != PredictorMode::GlobalOnly;
            const bool global_mode = cfg.predictor_mode != PredictorMode::BucketOnly;
            const uint64_t scan_now = now_ns();
            const bool global_near = global_mode && global_near_predicted_window(global_model, scan_now, cfg);
            if (bucket_mode) {
                fired_buckets.reserve(ws.size());
                for (size_t i = 0; i < ws.size(); ++i) {
                    const size_t bucket = bucket_for_line(ws[i], cfg);
                    if (near_predicted_window(models[bucket], scan_now, cfg)) {
                        should_prefetch[i] = static_cast<uint8_t>(std::min<size_t>(255, models[bucket].suggested_degree));
                        predicted_window = true;
                        fired_buckets.push_back(bucket);
                    }
                }
            }
            if (!predicted_window && global_near) {
                used_global_prediction = true;
                predicted_window = true;
                const size_t degree = std::max<size_t>(2, std::min<size_t>(cfg.prefetch_count, global_model.suggested_degree));
                for (size_t i = 0; i < ws.size(); ++i) should_prefetch[i] = static_cast<uint8_t>(degree);
                consume_prediction(global_model, scan_now);
            }
            if (predicted_window) {
                ++m.predicted_windows;
                if (used_global_prediction) ++m.global_predicted_windows;
                if (!used_global_prediction) {
                    std::sort(fired_buckets.begin(), fired_buckets.end());
                    fired_buckets.erase(std::unique(fired_buckets.begin(), fired_buckets.end()), fired_buckets.end());
                    for (size_t b : fired_buckets) consume_prediction(models[b], scan_now);
                }
                do_prefetch(buf, ws, should_prefetch, cfg, m);
            }
        }

        size_t outliers_this_iter = 0;
        std::vector<uint8_t> bucket_outlier(cfg.model_bucket_count, 0);
        std::vector<uint64_t> bucket_last_outlier_ns(cfg.model_bucket_count, 0);
        for (size_t i = 0; i < ws.size(); ++i) {
            const uint8_t* p = buf.line(ws[i]);
            const uint32_t cyc = timed_load(p);
            const uint64_t tns = now_ns();
            m.latencies.push_back(cyc);
            m.total_cycles += cyc;
            ++m.total_loads;
            if (cyc >= cfg.slow_threshold_cycles) ++m.slow_loads;
            if (cyc >= cfg.very_slow_cycles) ++m.very_slow_loads;
            if (cyc >= cfg.extreme_slow_cycles) ++m.extreme_slow_loads;
            if (refresh_aware_prefetch && should_prefetch[i]) ++m.prefetched_loads;

            const size_t bucket = bucket_for_line(ws[i], cfg);
            const bool outlier = is_outlier(latency_stats[bucket], cyc, cfg);
            const bool learning_event_bucket = is_learning_event(latency_stats[bucket], cyc, cfg);
            const bool learning_event_global = is_learning_event(global_latency_stats, cyc, cfg);
            if (learning_event_bucket) {
                ++m.learning_events;
                ++m.trained_event_count;
                update_model(models[bucket], tns, cfg);
            }
            if (learning_event_global) {
                ++m.learning_events;
                ++m.trained_event_count;
                update_model(global_model, tns, cfg);
            }
            update_latency_stats(latency_stats[bucket], cyc);
            update_latency_stats(global_latency_stats, cyc);
            if (outlier) {
                ++outliers_this_iter;
                bucket_outlier[bucket] = 1;
                bucket_last_outlier_ns[bucket] = tns;
            }
        }

        if (!training_phase) {
            m.total_outliers_in_eval += outliers_this_iter;
            const bool bad_window = outliers_this_iter >= cfg.min_bad_outliers_per_window;
            const bool severe_bad_window = outliers_this_iter >= (cfg.min_bad_outliers_per_window * 2);
            if (bad_window) ++m.actual_bad_windows;
            if (severe_bad_window) ++m.severe_bad_windows;
            if (predicted_window && !bad_window) ++m.useful_predicted_windows;
            if (bad_window && !predicted_window) ++m.missed_bad_windows;
            if (predicted_window && !bad_window) ++m.avoided_bad_windows;
            if (predicted_window && !bad_window) ++m.predicted_and_avoided_bad_windows;

            if (used_global_prediction) {
                apply_prediction_feedback(global_model, !bad_window, cfg);
                if (bad_window) {
                    uint64_t latest = 0;
                    for (uint64_t t : bucket_last_outlier_ns) latest = std::max(latest, t);
                    if (latest) apply_prediction_timing_feedback(global_model, latest, cfg);
                }
            } else if (!fired_buckets.empty()) {
                std::sort(fired_buckets.begin(), fired_buckets.end());
                fired_buckets.erase(std::unique(fired_buckets.begin(), fired_buckets.end()), fired_buckets.end());
                for (size_t b : fired_buckets) {
                    const bool useful = !bad_window || bucket_outlier[b] == 0;
                    apply_prediction_feedback(models[b], useful, cfg);
                    if (bucket_last_outlier_ns[b]) apply_prediction_timing_feedback(models[b], bucket_last_outlier_ns[b], cfg);
                }
            }
        }

        for (int spin = 0; spin < 32; ++spin) _mm_pause();
    }

    uint64_t active_count = 0;
    double conf_sum = 0.0, qual_sum = 0.0, degree_sum = 0.0;
    for (const auto& model : models) {
        if (!model.active) continue;
        ++active_count;
        conf_sum += model.confidence();
        qual_sum += model.quality();
        degree_sum += static_cast<double>(model.suggested_degree);
    }
    m.learned_buckets = active_count;
    if (active_count) {
        m.avg_active_bucket_confidence = conf_sum / active_count;
        m.avg_active_bucket_quality = qual_sum / active_count;
        m.avg_active_bucket_degree = degree_sum / active_count;
    }
    m.global_model_active = global_model.active;
    return m;
}

struct Summary {
    double avg = 0;
    uint32_t p50 = 0, p90 = 0, p95 = 0, p97 = 0, p99 = 0, p995 = 0, max = 0;
    std::array<uint64_t, 7> hist{};
};

Summary summarize(std::vector<uint32_t> vals) {
    if (vals.empty()) return {};
    std::sort(vals.begin(), vals.end());
    auto pick = [&](double q) {
        const size_t idx = static_cast<size_t>(q * (vals.size() - 1));
        return vals[idx];
    };
    Summary s;
    s.avg = static_cast<double>(std::accumulate(vals.begin(), vals.end(), uint64_t{0})) / vals.size();
    s.p50 = pick(0.50); s.p90 = pick(0.90); s.p95 = pick(0.95); s.p97 = pick(0.97);
    s.p99 = pick(0.99); s.p995 = pick(0.995); s.max = vals.back();
    for (uint32_t v : vals) {
        if (v < 64) ++s.hist[0];
        else if (v < 128) ++s.hist[1];
        else if (v < 192) ++s.hist[2];
        else if (v < 256) ++s.hist[3];
        else if (v < 320) ++s.hist[4];
        else if (v < 384) ++s.hist[5];
        else ++s.hist[6];
    }
    return s;
}

void print_report(const std::string& name, const Metrics& m, const Config& cfg, std::ostream& os) {
    Summary s = summarize(m.latencies);
    os << name << "\n";
    os << "  total_loads: " << m.total_loads << "\n";
    os << "  avg_cycles: " << std::fixed << std::setprecision(2) << s.avg << "\n";
    os << "  p50_cycles: " << s.p50 << "\n";
    os << "  p90_cycles: " << s.p90 << "\n";
    os << "  p95_cycles: " << s.p95 << "\n";
    os << "  p97_cycles: " << s.p97 << "\n";
    os << "  p99_cycles: " << s.p99 << "\n";
    os << "  p99_5_cycles: " << s.p995 << "\n";
    os << "  max_cycles: " << s.max << "\n";
    os << "  slow_loads: " << m.slow_loads << "\n";
    os << "  slow_ratio: " << std::setprecision(4)
       << (m.total_loads ? static_cast<double>(m.slow_loads) / m.total_loads : 0.0) << "\n";
    os << "  very_slow_loads: " << m.very_slow_loads << "\n";
    os << "  extreme_slow_loads: " << m.extreme_slow_loads << "\n";
    os << "  prefetch_instructions: " << m.prefetch_instructions << "\n";
    os << "  loads_in_predicted_windows: " << m.prefetched_loads << "\n";
    os << "  predicted_windows: " << m.predicted_windows << "\n";
    os << "  actual_bad_windows: " << m.actual_bad_windows << "\n";
    os << "  useful_predicted_windows: " << m.useful_predicted_windows << "\n";
    os << "  missed_bad_windows: " << m.missed_bad_windows << "\n";
    os << "  avoided_bad_windows: " << m.avoided_bad_windows << "\n";
    os << "  predicted_and_avoided_bad_windows: " << m.predicted_and_avoided_bad_windows << "\n";
    const double precision = m.predicted_windows ? static_cast<double>(m.useful_predicted_windows) / m.predicted_windows : 0.0;
    const double recall = m.actual_bad_windows ? static_cast<double>(m.useful_predicted_windows) / m.actual_bad_windows : 0.0;
    os << "  window_precision: " << std::setprecision(4) << precision << "\n";
    os << "  window_recall: " << std::setprecision(4) << recall << "\n";
    os << "  learned_buckets: " << m.learned_buckets << "\n";
    os << "  avg_active_bucket_confidence: " << std::setprecision(4) << m.avg_active_bucket_confidence << "\n";
    os << "  avg_active_bucket_quality: " << std::setprecision(4) << m.avg_active_bucket_quality << "\n";
    os << "  avg_active_bucket_degree: " << std::setprecision(2) << m.avg_active_bucket_degree << "\n";
    os << "  global_model_active: " << (m.global_model_active ? 1 : 0) << "\n";
    os << "  global_predicted_windows: " << m.global_predicted_windows << "\n";
    os << "  learning_events: " << m.learning_events << "\n";
    os << "  trained_event_count: " << m.trained_event_count << "\n";
    os << "  severe_bad_windows: " << m.severe_bad_windows << "\n";
    os << "  avg_outliers_per_eval_window: " << std::setprecision(2)
       << (m.eval_iterations ? static_cast<double>(m.total_outliers_in_eval) / m.eval_iterations : 0.0) << "\n";
    os << "  training_iterations: " << m.training_iterations << "\n";
    os << "  eval_iterations: " << m.eval_iterations << "\n";
    os << "  flushes: " << m.flushes << "\n";
    os << "  histogram_lt64: " << s.hist[0] << "\n";
    os << "  histogram_64_127: " << s.hist[1] << "\n";
    os << "  histogram_128_191: " << s.hist[2] << "\n";
    os << "  histogram_192_255: " << s.hist[3] << "\n";
    os << "  histogram_256_319: " << s.hist[4] << "\n";
    os << "  histogram_320_383: " << s.hist[5] << "\n";
    os << "  histogram_384_plus: " << s.hist[6] << "\n";
}

Config parse_args(int argc, char** argv, std::string& report_path) {
    Config cfg;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&](size_t& out) { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); out = static_cast<size_t>(std::stoull(argv[++i])); };
        auto next_u64 = [&](uint64_t& out) { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); out = static_cast<uint64_t>(std::stoull(argv[++i])); };
        if (a == "--lines") next(cfg.lines);
        else if (a == "--working-set") next(cfg.working_set);
        else if (a == "--iterations") next(cfg.iterations);
        else if (a == "--prefetch-count") next(cfg.prefetch_count);
        else if (a == "--lead-ns") next(cfg.lead_ns);
        else if (a == "--refresh-guard-ns") next(cfg.refresh_guard_ns);
        else if (a == "--eviction-stride") next(cfg.eviction_stride);
        else if (a == "--slow-threshold-cycles") next_u64(cfg.slow_threshold_cycles);
        else if (a == "--model-buckets") next(cfg.model_bucket_count);
        else if (a == "--min-samples") next(cfg.min_samples);
        else if (a == "--min-period-ns") next(cfg.min_period_ns);
        else if (a == "--max-period-ns") next(cfg.max_period_ns);
        else if (a == "--period-tolerance-pct") next(cfg.period_tolerance_pct);
        else if (a == "--confidence-threshold-pct") next(cfg.confidence_threshold_pct);
        else if (a == "--training-pct") next(cfg.training_pct);
        else if (a == "--min-outlier-cycles") next_u64(cfg.min_outlier_cycles);
        else if (a == "--outlier-sigma-x100") next(cfg.outlier_sigma_x100);
        else if (a == "--min-dev-cycles") next_u64(cfg.min_dev_cycles);
        else if (a == "--learn-min-slow-cycles") next_u64(cfg.learn_min_slow_cycles);
        else if (a == "--learn-sigma-x100") next(cfg.learn_sigma_x100);
        else if (a == "--learn-min-dev-cycles") next_u64(cfg.learn_min_dev_cycles);
        else if (a == "--min-bad-outliers-per-window") next(cfg.min_bad_outliers_per_window);
        else if (a == "--very-slow-cycles") next_u64(cfg.very_slow_cycles);
        else if (a == "--extreme-slow-cycles") next_u64(cfg.extreme_slow_cycles);
        else if (a == "--predictor-mode") {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            const std::string v = argv[++i];
            if (v == "hybrid") cfg.predictor_mode = PredictorMode::Hybrid;
            else if (v == "bucket-only") cfg.predictor_mode = PredictorMode::BucketOnly;
            else if (v == "global-only") cfg.predictor_mode = PredictorMode::GlobalOnly;
            else throw std::runtime_error("predictor-mode must be hybrid, bucket-only, or global-only");
        }
        else if (a == "--global-confidence-threshold-pct") next(cfg.global_confidence_threshold_pct);
        else if (a == "--global-cooldown-pct") next(cfg.global_cooldown_pct);
        else if (a == "--seed") cfg.seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (a == "--report") report_path = argv[++i];
        else if (a == "--hugepage-hint") cfg.use_hugepage_hint = true;
        else if (a == "--help") {
            std::cout << "See source/config fields for options.\n";
            std::exit(0);
        }
        else throw std::runtime_error("unknown argument: " + a);
    }
    if (cfg.lines == 0 || cfg.working_set == 0 || cfg.working_set > cfg.lines) throw std::runtime_error("working set must be > 0 and <= total lines");
    if (cfg.model_bucket_count == 0 || (cfg.model_bucket_count & (cfg.model_bucket_count - 1)) != 0) throw std::runtime_error("model bucket count must be a power of two");
    if (cfg.min_period_ns > cfg.max_period_ns) throw std::runtime_error("min-period-ns must be <= max-period-ns");
    if (cfg.training_pct > 100) throw std::runtime_error("training-pct must be between 0 and 100");
    if (cfg.global_confidence_threshold_pct > 100 || cfg.global_cooldown_pct > 1000) throw std::runtime_error("global thresholds out of range");
    return cfg;
}

} // namespace

int main(int argc, char** argv) {
    try {
        std::string report_path;
        Config cfg = parse_args(argc, argv, report_path);
        Buffer buf(cfg.lines);
        touch_buffer(buf);

        auto baseline = run_case(buf, cfg, false);
        auto pref = run_case(buf, cfg, true);

        std::ostream* out = &std::cout;
        std::ofstream file;
        if (!report_path.empty()) {
            file.open(report_path);
            out = &file;
        }

        (*out) << "Real-memory learned-window prefetch PoC\n";
        (*out) << "Configuration\n";
        (*out) << "  lines: " << cfg.lines << "\n";
        (*out) << "  working_set: " << cfg.working_set << "\n";
        (*out) << "  iterations: " << cfg.iterations << "\n";
        (*out) << "  prefetch_count: " << cfg.prefetch_count << "\n";
        (*out) << "  lead_ns: " << cfg.lead_ns << "\n";
        (*out) << "  refresh_guard_ns: " << cfg.refresh_guard_ns << "\n";
        (*out) << "  eviction_stride: " << cfg.eviction_stride << "\n";
        (*out) << "  slow_threshold_cycles: " << cfg.slow_threshold_cycles << "\n";
        (*out) << "  model_bucket_count: " << cfg.model_bucket_count << "\n";
        (*out) << "  min_samples: " << cfg.min_samples << "\n";
        (*out) << "  min_period_ns: " << cfg.min_period_ns << "\n";
        (*out) << "  max_period_ns: " << cfg.max_period_ns << "\n";
        (*out) << "  period_tolerance_pct: " << cfg.period_tolerance_pct << "\n";
        (*out) << "  confidence_threshold_pct: " << cfg.confidence_threshold_pct << "\n";
        (*out) << "  training_pct: " << cfg.training_pct << "\n";
        (*out) << "  min_outlier_cycles: " << cfg.min_outlier_cycles << "\n";
        (*out) << "  outlier_sigma_x100: " << cfg.outlier_sigma_x100 << "\n";
        (*out) << "  min_dev_cycles: " << cfg.min_dev_cycles << "\n";
        (*out) << "  learn_min_slow_cycles: " << cfg.learn_min_slow_cycles << "\n";
        (*out) << "  learn_sigma_x100: " << cfg.learn_sigma_x100 << "\n";
        (*out) << "  learn_min_dev_cycles: " << cfg.learn_min_dev_cycles << "\n";
        (*out) << "  min_bad_outliers_per_window: " << cfg.min_bad_outliers_per_window << "\n";
        (*out) << "  very_slow_cycles: " << cfg.very_slow_cycles << "\n";
        (*out) << "  extreme_slow_cycles: " << cfg.extreme_slow_cycles << "\n";
        (*out) << "  predictor_mode: " << (cfg.predictor_mode == PredictorMode::Hybrid ? "hybrid" : (cfg.predictor_mode == PredictorMode::BucketOnly ? "bucket-only" : "global-only")) << "\n";
        (*out) << "  global_confidence_threshold_pct: " << cfg.global_confidence_threshold_pct << "\n";
        (*out) << "  global_cooldown_pct: " << cfg.global_cooldown_pct << "\n\n";

        print_report("Baseline", baseline, cfg, *out);
        (*out) << "\n";
        print_report("RefreshAwarePrefetch", pref, cfg, *out);
        (*out) << "\nComparison\n";
        const double avg_reduction = baseline.total_cycles ? (static_cast<double>(static_cast<int64_t>(baseline.total_cycles) - static_cast<int64_t>(pref.total_cycles)) / baseline.total_cycles) * 100.0 : 0.0;
        const double slow_reduction = baseline.slow_loads ? (static_cast<double>(static_cast<int64_t>(baseline.slow_loads) - static_cast<int64_t>(pref.slow_loads)) / baseline.slow_loads) * 100.0 : 0.0;
        const double very_slow_reduction = baseline.very_slow_loads ? (static_cast<double>(static_cast<int64_t>(baseline.very_slow_loads) - static_cast<int64_t>(pref.very_slow_loads)) / baseline.very_slow_loads) * 100.0 : 0.0;
        const double extreme_slow_reduction = baseline.extreme_slow_loads ? (static_cast<double>(static_cast<int64_t>(baseline.extreme_slow_loads) - static_cast<int64_t>(pref.extreme_slow_loads)) / baseline.extreme_slow_loads) * 100.0 : 0.0;
        const auto bs = summarize(baseline.latencies);
        const auto ps = summarize(pref.latencies);
        (*out) << std::fixed << std::setprecision(2);
        (*out) << "  avg_cycle_reduction_pct: " << avg_reduction << "\n";
        (*out) << "  slow_load_reduction_pct: " << slow_reduction << "\n";
        (*out) << "  very_slow_reduction_pct: " << very_slow_reduction << "\n";
        (*out) << "  extreme_slow_reduction_pct: " << extreme_slow_reduction << "\n";
        (*out) << "  p95_delta_cycles: " << static_cast<int>(bs.p95) - static_cast<int>(ps.p95) << "\n";
        (*out) << "  p99_delta_cycles: " << static_cast<int>(bs.p99) - static_cast<int>(ps.p99) << "\n";
        (*out) << "  p99_5_delta_cycles: " << static_cast<int>(bs.p995) - static_cast<int>(ps.p995) << "\n";
        (*out) << "\nNotes\n";
        (*out) << "  - This uses real loads, clflush-based eviction, rdtscp timing, and software prefetch hints.\n";
        (*out) << "  - It still does NOT observe real DRAM refresh commands from user space.\n";
        (*out) << "  - The bad window is learned from repeated slow-read intervals per hashed address bucket, with a global cadence fallback.\n";
        (*out) << "  - Training and evaluation are separated to reduce self-justifying predictions.\n";
    } catch (const std::exception& ex) {
        std::cerr << "error: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
