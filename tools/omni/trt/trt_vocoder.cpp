#include "trt_vocoder.h"

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>

#define TRT_CUDA_CHECK(call)                                                  \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "[TRT] CUDA error at %s:%d: %s (%s)\n",     \
                         __FILE__, __LINE__, cudaGetErrorString(err_), #call);\
            return false;                                                     \
        }                                                                     \
    } while (0)

#define TRT_CUDA_CHECK_WARN(call)                                             \
    do {                                                                      \
        cudaError_t err_ = (call);                                            \
        if (err_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "[TRT] CUDA error at %s:%d: %s (%s)\n",     \
                         __FILE__, __LINE__, cudaGetErrorString(err_), #call);\
        }                                                                     \
    } while (0)

#define TRT_CHECK(call, msg)                                                  \
    do {                                                                      \
        if (!(call)) {                                                        \
            std::fprintf(stderr, "[TRT] %s failed at %s:%d\n",               \
                         msg, __FILE__, __LINE__);                            \
            return false;                                                     \
        }                                                                     \
    } while (0)

namespace omni {
namespace vocoder {

static constexpr int kNfft     = 16;
static constexpr int kHop      = 4;
static constexpr int kNumFreqs = 9;
static constexpr int kStftCh   = 18;
static constexpr int kPad      = 8;
static constexpr int kMelCh    = 80;

namespace {
class TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char * msg) noexcept override {
        if (severity <= Severity::kWARNING)
            std::fprintf(stderr, "[TRT] %s\n", msg);
    }
};
static TrtLogger g_logger;
} // namespace

TrtVocoder::TrtVocoder() = default;

TrtVocoder::~TrtVocoder() {
    if (d_mel_)  TRT_CUDA_CHECK_WARN(cudaFree(d_mel_));
    if (d_stft_) TRT_CUDA_CHECK_WARN(cudaFree(d_stft_));
    if (stream_) TRT_CUDA_CHECK_WARN(cudaStreamDestroy(stream_));
    delete ctx_;
    delete engine_;
    delete runtime_;
}

bool TrtVocoder::init(const TrtVocoderConfig & cfg) {
    cfg_ = cfg;

    std::ifstream f(cfg_.engine_path, std::ios::binary);
    if (!f) {
        std::fprintf(stderr, "[TRT] cannot open engine: %s\n", cfg_.engine_path.c_str());
        return false;
    }
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> plan(sz);
    f.read(plan.data(), sz);
    std::fprintf(stderr, "[TRT] loaded engine: %s (%.1f MB)\n",
                 cfg_.engine_path.c_str(), sz / 1e6);

    runtime_ = nvinfer1::createInferRuntime(g_logger);
    if (!runtime_) {
        std::fprintf(stderr, "[TRT] createInferRuntime failed\n");
        return false;
    }

    engine_ = runtime_->deserializeCudaEngine(plan.data(), sz);
    if (!engine_) {
        std::fprintf(stderr, "[TRT] deserializeCudaEngine failed\n");
        return false;
    }

    ctx_ = engine_->createExecutionContext();
    if (!ctx_) {
        std::fprintf(stderr, "[TRT] createExecutionContext failed\n");
        return false;
    }

    nvinfer1::Dims stft_dims = engine_->getTensorShape("stft_18ch");
    if (stft_dims.nbDims != 3 || stft_dims.d[1] != kStftCh) {
        std::fprintf(stderr, "[TRT] unexpected stft_18ch shape: [%d,%d,%d]\n",
                     stft_dims.d[0], stft_dims.d[1], stft_dims.d[2]);
        return false;
    }
    T_frame_    = stft_dims.d[2];
    stft_bytes_ = 1 * kStftCh * T_frame_ * sizeof(float);

    nvinfer1::Dims mel_dims;
    mel_dims.nbDims = 3;
    mel_dims.d[0]   = 1;
    mel_dims.d[1]   = kMelCh;
    mel_dims.d[2]   = cfg_.max_mel_frames;
    if (!ctx_->setInputShape("mel", mel_dims)) {
        std::fprintf(stderr, "[TRT] setInputShape failed\n");
        return false;
    }

    TRT_CUDA_CHECK(cudaStreamCreate(&stream_));
    mel_bytes_ = 1 * kMelCh * cfg_.max_mel_frames * sizeof(float);
    TRT_CUDA_CHECK(cudaMalloc(&d_mel_,  mel_bytes_));
    TRT_CUDA_CHECK(cudaMalloc(&d_stft_, stft_bytes_));

    hann_window_.resize(kNfft);
    for (int i = 0; i < kNfft; ++i)
        hann_window_[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (kNfft - 1)));

    idft_matrix_.resize(kNfft * kStftCh);
    for (int n = 0; n < kNfft; ++n) {
        for (int k = 0; k < kNumFreqs; ++k) {
            float angle = 2.0f * M_PI * (k + 1) * n / (float)kNfft;
            idft_matrix_[n * kStftCh + k]             =  std::cos(angle) * 2.0f / kNfft;
            idft_matrix_[n * kStftCh + kNumFreqs + k] = -std::sin(angle) * 2.0f / kNfft;
        }
    }

    ready_ = true;
    std::fprintf(stderr, "[TRT] vocoder ready, max_mel_frames=%d, T_frame=%d\n",
                 cfg_.max_mel_frames, T_frame_);
    return true;
}

bool TrtVocoder::run(const float * mel, int n_mel_frames,
                     std::vector<float> & wave_out, int64_t & n_samples) {
    if (!ready_) return false;

    int T_pad = cfg_.max_mel_frames;
    std::vector<float> padded(kMelCh * T_pad, 0.0f);
    int T_copy = std::min(n_mel_frames, T_pad);
    for (int c = 0; c < kMelCh; ++c)
        for (int t = 0; t < T_copy; ++t)
            padded[c * T_pad + t] = mel[t * kMelCh + c];

    TRT_CUDA_CHECK(cudaMemcpyAsync(d_mel_, padded.data(), mel_bytes_,
                                    cudaMemcpyHostToDevice, stream_));

    ctx_->setTensorAddress("mel",       d_mel_);
    ctx_->setTensorAddress("stft_18ch", d_stft_);
    TRT_CHECK(ctx_->enqueueV3(stream_), "enqueueV3");
    TRT_CUDA_CHECK(cudaStreamSynchronize(stream_));

    std::vector<float> stft(kStftCh * T_frame_);
    TRT_CUDA_CHECK(cudaMemcpy(stft.data(), d_stft_, stft_bytes_, cudaMemcpyDeviceToHost));

    int audio_len = (T_frame_ - 1) * kHop + kNfft;
    wave_out.assign(audio_len, 0.0f);

    for (int frame = 0; frame < T_frame_; ++frame) {
        float stft_vals[kStftCh];
        for (int c = 0; c < kStftCh; ++c)
            stft_vals[c] = stft[c * T_frame_ + frame];

        int base = frame * kHop;
        for (int n = 0; n < kNfft; ++n) {
            float v = 0;
            const float * row = &idft_matrix_[n * kStftCh];
            for (int k = 0; k < kStftCh; ++k)
                v += row[k] * stft_vals[k];
            wave_out[base + n] += v * hann_window_[n];
        }
    }

    n_samples = audio_len - 2 * kPad;
    if (n_samples < 0) n_samples = 0;
    return true;
}

} // namespace vocoder
} // namespace omni
