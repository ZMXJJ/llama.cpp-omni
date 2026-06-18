#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}
typedef struct CUstream_st *cudaStream_t;

namespace omni {
namespace vocoder {

struct TrtVocoderConfig {
    std::string engine_path;
    int         max_mel_frames = 100;
};

class TrtVocoder {
public:
    TrtVocoder();
    ~TrtVocoder();

    bool init(const TrtVocoderConfig & cfg);
    bool ready() const { return ready_; }

    bool run(const float * mel, int n_mel_frames,
             std::vector<float> & wave_out, int64_t & n_samples);

private:
    bool             ready_ = false;
    TrtVocoderConfig cfg_;

    nvinfer1::IRuntime          * runtime_ = nullptr;
    nvinfer1::ICudaEngine       * engine_  = nullptr;
    nvinfer1::IExecutionContext * ctx_     = nullptr;
    cudaStream_t                  stream_  = nullptr;

    void * d_mel_      = nullptr;
    void * d_stft_     = nullptr;
    size_t mel_bytes_  = 0;
    size_t stft_bytes_ = 0;
    int    T_frame_    = 0;

    std::vector<float> hann_window_;
    std::vector<float> idft_matrix_;
};

} // namespace vocoder
} // namespace omni
