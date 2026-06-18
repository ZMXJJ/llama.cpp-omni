// Standalone TRT vocoder benchmark.
// Build: cmake -DUSE_TRT_VOCODER=ON ... && cmake --build . --target llama-omni-test-trt-vocoder
// Run:   OMNI_TRT_VOCODER_ENGINE=/path/to/vocoder.plan ./bin/llama-omni-test-trt-vocoder

#include "trt/trt_vocoder.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

int main() {
    const char * engine_path = std::getenv("OMNI_TRT_VOCODER_ENGINE");
    if (!engine_path || !engine_path[0]) {
        std::fprintf(stderr, "Usage: OMNI_TRT_VOCODER_ENGINE=<path> %s\n", "llama-omni-test-trt-vocoder");
        return 1;
    }

    omni::vocoder::TrtVocoderConfig cfg;
    cfg.engine_path    = engine_path;
    cfg.max_mel_frames = 100;

    omni::vocoder::TrtVocoder vocoder;
    if (!vocoder.init(cfg)) {
        std::fprintf(stderr, "TrtVocoder init failed\n");
        return 1;
    }

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> mel(80 * cfg.max_mel_frames);
    for (auto & v : mel) v = dist(rng);

    // Warmup
    std::vector<float> wave;
    int64_t n_samples = 0;
    if (!vocoder.run(mel.data(), cfg.max_mel_frames, wave, n_samples)) {
        std::fprintf(stderr, "warmup run failed\n");
        return 1;
    }

    // Benchmark
    constexpr int kIters = 100;
    double total_ms = 0.0;
    double min_ms   = 1e9;
    double max_ms   = 0.0;

    for (int i = 0; i < kIters; ++i) {
        for (auto & v : mel) v = dist(rng);

        auto t0 = std::chrono::steady_clock::now();
        if (!vocoder.run(mel.data(), cfg.max_mel_frames, wave, n_samples)) {
            std::fprintf(stderr, "run failed at iter %d\n", i);
            return 1;
        }
        auto t1 = std::chrono::steady_clock::now();

        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
    }

    double avg_ms = total_ms / kIters;
    double dur_s  = n_samples / 16000.0;
    double rtf    = (avg_ms / 1000.0) / dur_s;

    std::fprintf(stderr, "=== TRT Vocoder Benchmark ===\n");
    std::fprintf(stderr, "  Engine:      %s\n",   engine_path);
    std::fprintf(stderr, "  T_mel:       %d\n",   cfg.max_mel_frames);
    std::fprintf(stderr, "  Audio len:   %lld samples (%.3f s @ 16kHz)\n",
                 (long long)n_samples, dur_s);
    std::fprintf(stderr, "  Iterations:  %d\n",   kIters);
    std::fprintf(stderr, "  Min:         %.3f ms\n", min_ms);
    std::fprintf(stderr, "  Max:         %.3f ms\n", max_ms);
    std::fprintf(stderr, "  Avg:         %.3f ms\n", avg_ms);
    std::fprintf(stderr, "  RTF:         %.4f\n",  rtf);
    std::fprintf(stderr, "  Wave range:  [%.6f, %.6f]\n",
                 *std::min_element(wave.begin(), wave.end()),
                 *std::max_element(wave.begin(), wave.end()));

    return 0;
}
