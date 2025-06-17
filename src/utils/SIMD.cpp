/**
 * @file SIMD.cpp
 * @brief SIMD optimized operations for high-performance computing
 * @version 1.0.0
 * @date 2025-05-25
 */

#include "LwTT/utils/SIMD.hpp"
#include <cstring>
#include <algorithm>
#include <stdexcept>

// Platform-specific SIMD includes
#ifdef __AVX512F__
#include <immintrin.h>
#define SIMD_WIDTH 16  // 512-bit / 32-bit = 16 floats
#define HAS_AVX512
#elif defined(__AVX2__) || defined(__AVX__)
#include <immintrin.h>
#define SIMD_WIDTH 8   // 256-bit / 32-bit = 8 floats
#define HAS_AVX
#elif defined(__SSE4_1__) || defined(__SSE2__)
#include <emmintrin.h>
#include <smmintrin.h>
#define SIMD_WIDTH 4   // 128-bit / 32-bit = 4 floats
#define HAS_SSE
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define SIMD_WIDTH 4   // 128-bit / 32-bit = 4 floats
#define HAS_NEON
#else
#define SIMD_WIDTH 1   // No SIMD, fallback to scalar
#endif

namespace crllwtt {
namespace Utils {

// Static constants
const int SIMDUtils::kSIMDWidth = SIMD_WIDTH;

// Helper function to check alignment
inline bool IsAligned(const void* ptr, size_t alignment) {
    return (reinterpret_cast<uintptr_t>(ptr) % alignment) == 0;
}

void SIMDUtils::VectorAdd(const float* a, const float* b, float* result, int size) {
    int i = 0;
    
#ifdef HAS_AVX512
    // AVX-512 implementation
    const int simd_end = (size / 16) * 16;
    for (; i < simd_end; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        __m512 vr = _mm512_add_ps(va, vb);
        _mm512_storeu_ps(&result[i], vr);
    }
#elif defined(HAS_AVX)
    // AVX/AVX2 implementation
    const int simd_end = (size / 8) * 8;
    for (; i < simd_end; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_add_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }
#elif defined(HAS_SSE)
    // SSE implementation
    const int simd_end = (size / 4) * 4;
    for (; i < simd_end; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vr = _mm_add_ps(va, vb);
        _mm_storeu_ps(&result[i], vr);
    }
#elif defined(HAS_NEON)
    // ARM NEON implementation
    const int simd_end = (size / 4) * 4;
    for (; i < simd_end; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vr = vaddq_f32(va, vb);
        vst1q_f32(&result[i], vr);
    }
#endif
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = a[i] + b[i];
    }
}

void SIMDUtils::VectorMul(const float* a, const float* b, float* result, int size) {
    int i = 0;
    
#ifdef HAS_AVX512
    const int simd_end = (size / 16) * 16;
    for (; i < simd_end; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        __m512 vr = _mm512_mul_ps(va, vb);
        _mm512_storeu_ps(&result[i], vr);
    }
#elif defined(HAS_AVX)
    const int simd_end = (size / 8) * 8;
    for (; i < simd_end; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vr = _mm256_mul_ps(va, vb);
        _mm256_storeu_ps(&result[i], vr);
    }
#elif defined(HAS_SSE)
    const int simd_end = (size / 4) * 4;
    for (; i < simd_end; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vr = _mm_mul_ps(va, vb);
        _mm_storeu_ps(&result[i], vr);
    }
#elif defined(HAS_NEON)
    const int simd_end = (size / 4) * 4;
    for (; i < simd_end; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        float32x4_t vr = vmulq_f32(va, vb);
        vst1q_f32(&result[i], vr);
    }
#endif
    
    // Handle remaining elements
    for (; i < size; ++i) {
        result[i] = a[i] * b[i];
    }
}

float SIMDUtils::DotProduct(const float* a, const float* b, int size) {
    float sum = 0.0f;
    int i = 0;
    
#ifdef HAS_AVX512
    __m512 sum_vec = _mm512_setzero_ps();
    const int simd_end = (size / 16) * 16;
    for (; i < simd_end; i += 16) {
        __m512 va = _mm512_loadu_ps(&a[i]);
        __m512 vb = _mm512_loadu_ps(&b[i]);
        sum_vec = _mm512_fmadd_ps(va, vb, sum_vec);
    }
    // Horizontal sum
    sum = _mm512_reduce_add_ps(sum_vec);
#elif defined(HAS_AVX)
    __m256 sum_vec = _mm256_setzero_ps();
    const int simd_end = (size / 8) * 8;
    for (; i < simd_end; i += 8) {
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        #ifdef __FMA__
        sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);
        #else
        __m256 vr = _mm256_mul_ps(va, vb);
        sum_vec = _mm256_add_ps(sum_vec, vr);
        #endif
    }
    // Horizontal sum
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
    __m128 sum_128 = _mm_add_ps(sum_high, sum_low);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum = _mm_cvtss_f32(sum_128);
#elif defined(HAS_SSE)
    __m128 sum_vec = _mm_setzero_ps();
    const int simd_end = (size / 4) * 4;
    for (; i < simd_end; i += 4) {
        __m128 va = _mm_loadu_ps(&a[i]);
        __m128 vb = _mm_loadu_ps(&b[i]);
        __m128 vr = _mm_mul_ps(va, vb);
        sum_vec = _mm_add_ps(sum_vec, vr);
    }
    // Horizontal sum
    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum_vec = _mm_hadd_ps(sum_vec, sum_vec);
    sum = _mm_cvtss_f32(sum_vec);
#elif defined(HAS_NEON)
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    const int simd_end = (size / 4) * 4;
    for (; i < simd_end; i += 4) {
        float32x4_t va = vld1q_f32(&a[i]);
        float32x4_t vb = vld1q_f32(&b[i]);
        sum_vec = vfmaq_f32(sum_vec, va, vb);
    }
    // Horizontal sum
    float32x2_t sum_pair = vadd_f32(vget_high_f32(sum_vec), vget_low_f32(sum_vec));
    sum = vget_lane_f32(vpadd_f32(sum_pair, sum_pair), 0);
#endif
    
    // Handle remaining elements
    for (; i < size; ++i) {
        sum += a[i] * b[i];
    }
    
    return sum;
}

void SIMDUtils::MatrixMultiply(const float* a, const float* b, float* c,
                              int m, int n, int k,
                              bool transpose_a, bool transpose_b) {
    // Simple implementation - can be further optimized with blocking and register tiling
    if (!transpose_a && !transpose_b) {
        // C = A * B, A: m×k, B: k×n, C: m×n
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                [[maybe_unused]] int vec_end = (k / SIMD_WIDTH) * SIMD_WIDTH;
                int l = 0;
                
                // Vectorized dot product
                #ifdef HAS_AVX512
                __m512 sum_vec = _mm512_setzero_ps();
                for (; l < vec_end; l += 16) {
                    __m512 va = _mm512_loadu_ps(&a[i * k + l]);
                    __m512 vb = _mm512_loadu_ps(&b[l * n + j]);
                    sum_vec = _mm512_fmadd_ps(va, vb, sum_vec);
                }
                sum = _mm512_reduce_add_ps(sum_vec);
                #elif defined(HAS_AVX)
                __m256 sum_vec = _mm256_setzero_ps();
                for (; l < vec_end; l += 8) {
                    __m256 va = _mm256_loadu_ps(&a[i * k + l]);
                    // Need to gather elements for B since they're not contiguous
                    __m256 vb = _mm256_set_ps(
                        b[(l+7)*n + j], b[(l+6)*n + j], b[(l+5)*n + j], b[(l+4)*n + j],
                        b[(l+3)*n + j], b[(l+2)*n + j], b[(l+1)*n + j], b[l*n + j]
                    );
                    #ifdef __FMA__
                    sum_vec = _mm256_fmadd_ps(va, vb, sum_vec);
                    #else
                    __m256 vr = _mm256_mul_ps(va, vb);
                    sum_vec = _mm256_add_ps(sum_vec, vr);
                    #endif
                }
                // Horizontal sum
                __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
                __m128 sum_low = _mm256_castps256_ps128(sum_vec);
                __m128 sum_128 = _mm_add_ps(sum_high, sum_low);
                sum_128 = _mm_hadd_ps(sum_128, sum_128);
                sum_128 = _mm_hadd_ps(sum_128, sum_128);
                sum += _mm_cvtss_f32(sum_128);
                #endif
                
                // Handle remaining elements
                for (; l < k; ++l) {
                    sum += a[i * k + l] * b[l * n + j];
                }
                
                c[i * n + j] = sum;
            }
        }
    } else {
        // Handle transposed cases with standard loops for now
        // TODO: Optimize transposed cases with SIMD
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int l = 0; l < k; ++l) {
                    float a_val = transpose_a ? a[l * m + i] : a[i * k + l];
                    float b_val = transpose_b ? b[j * k + l] : b[l * n + j];
                    sum += a_val * b_val;
                }
                c[i * n + j] = sum;
            }
        }
    }
}

void SIMDUtils::ApplyActivation(const float* input, float* output, int size, ActivationType activation) {
    int i = 0;
    
    switch (activation) {
        case ActivationType::ReLU: {
            #ifdef HAS_AVX512
            __m512 zero = _mm512_setzero_ps();
            const int simd_end = (size / 16) * 16;
            for (; i < simd_end; i += 16) {
                __m512 x = _mm512_loadu_ps(&input[i]);
                __m512 result = _mm512_max_ps(x, zero);
                _mm512_storeu_ps(&output[i], result);
            }
            #elif defined(HAS_AVX)
            __m256 zero = _mm256_setzero_ps();
            const int simd_end = (size / 8) * 8;
            for (; i < simd_end; i += 8) {
                __m256 x = _mm256_loadu_ps(&input[i]);
                __m256 result = _mm256_max_ps(x, zero);
                _mm256_storeu_ps(&output[i], result);
            }
            #elif defined(HAS_SSE)
            __m128 zero = _mm_setzero_ps();
            const int simd_end = (size / 4) * 4;
            for (; i < simd_end; i += 4) {
                __m128 x = _mm_loadu_ps(&input[i]);
                __m128 result = _mm_max_ps(x, zero);
                _mm_storeu_ps(&output[i], result);
            }
            #elif defined(HAS_NEON)
            float32x4_t zero = vdupq_n_f32(0.0f);
            const int simd_end = (size / 4) * 4;
            for (; i < simd_end; i += 4) {
                float32x4_t x = vld1q_f32(&input[i]);
                float32x4_t result = vmaxq_f32(x, zero);
                vst1q_f32(&output[i], result);
            }
            #endif
            
            // Handle remaining elements
            for (; i < size; ++i) {
                output[i] = std::max(0.0f, input[i]);
            }
            break;
        }
        
        case ActivationType::Tanh: {
            // Tanh is more complex, use scalar for now
            // TODO: Implement SIMD version with approximation
            for (int j = 0; j < size; ++j) {
                output[j] = std::tanh(input[j]);
            }
            break;
        }
        
        case ActivationType::Sigmoid: {
            // Sigmoid is complex, use scalar for now
            // TODO: Implement SIMD version with approximation
            for (int j = 0; j < size; ++j) {
                output[j] = 1.0f / (1.0f + std::exp(-input[j]));
            }
            break;
        }
        
        case ActivationType::GELU: {
            // GELU is complex, use scalar for now
            // TODO: Implement SIMD version with approximation
            const float sqrt_2_pi = std::sqrt(2.0f / M_PI);
            for (int j = 0; j < size; ++j) {
                float x = input[j];
                output[j] = 0.5f * x * (1.0f + std::tanh(sqrt_2_pi * (x + 0.044715f * x * x * x)));
            }
            break;
        }
    }
}

void SIMDUtils::SoftMax(const float* input, float* output, int size) {
    // Find maximum for numerical stability
    float max_val = input[0];
    for (int i = 1; i < size; ++i) {
        max_val = std::max(max_val, input[i]);
    }
    
    // Compute exp(x - max) and sum
    float sum = 0.0f;
    int i = 0;
    
    #ifdef HAS_AVX512
    __m512 max_vec = _mm512_set1_ps(max_val);
    __m512 sum_vec = _mm512_setzero_ps();
    const int simd_end = (size / 16) * 16;
    for (; i < simd_end; i += 16) {
        __m512 x = _mm512_loadu_ps(&input[i]);
        __m512 x_shifted = _mm512_sub_ps(x, max_vec);
        // Use approximation for exp - exact implementation would need a more complex approach
        __m512 exp_x = _mm512_exp_ps(x_shifted); // This intrinsic may not exist on all platforms
        _mm512_storeu_ps(&output[i], exp_x);
        sum_vec = _mm512_add_ps(sum_vec, exp_x);
    }
    sum = _mm512_reduce_add_ps(sum_vec);
    #endif
    
    // Handle remaining elements and compute sum
    for (; i < size; ++i) {
        output[i] = std::exp(input[i] - max_val);
        sum += output[i];
    }
    
    // Normalize
    if (sum > 0.0f) {
        float inv_sum = 1.0f / sum;
        VectorScale(output, inv_sum, output, size);
    }
}

void SIMDUtils::VectorScale(const float* input, float scale, float* output, int size) {
    int i = 0;
    
    #ifdef HAS_AVX512
    __m512 scale_vec = _mm512_set1_ps(scale);
    const int simd_end = (size / 16) * 16;
    for (; i < simd_end; i += 16) {
        __m512 x = _mm512_loadu_ps(&input[i]);
        __m512 result = _mm512_mul_ps(x, scale_vec);
        _mm512_storeu_ps(&output[i], result);
    }
    #elif defined(HAS_AVX)
    __m256 scale_vec = _mm256_set1_ps(scale);
    const int simd_end = (size / 8) * 8;
    for (; i < simd_end; i += 8) {
        __m256 x = _mm256_loadu_ps(&input[i]);
        __m256 result = _mm256_mul_ps(x, scale_vec);
        _mm256_storeu_ps(&output[i], result);
    }
    #elif defined(HAS_SSE)
    __m128 scale_vec = _mm_set1_ps(scale);
    const int simd_end = (size / 4) * 4;
    for (; i < simd_end; i += 4) {
        __m128 x = _mm_loadu_ps(&input[i]);
        __m128 result = _mm_mul_ps(x, scale_vec);
        _mm_storeu_ps(&output[i], result);
    }
    #elif defined(HAS_NEON)
    float32x4_t scale_vec = vdupq_n_f32(scale);
    const int simd_end = (size / 4) * 4;
    for (; i < simd_end; i += 4) {
        float32x4_t x = vld1q_f32(&input[i]);
        float32x4_t result = vmulq_f32(x, scale_vec);
        vst1q_f32(&output[i], result);
    }
    #endif
    
    // Handle remaining elements
    for (; i < size; ++i) {
        output[i] = input[i] * scale;
    }
}

std::string SIMDUtils::GetSIMDInfo() {
    std::string info = "SIMD Support: ";
    
    #ifdef HAS_AVX512
    info += "AVX-512 ";
    #endif
    #ifdef HAS_AVX
    info += "AVX ";
    #endif
    #ifdef HAS_SSE
    info += "SSE ";
    #endif
    #ifdef HAS_NEON
    info += "NEON ";
    #endif
    
    if (info == "SIMD Support: ") {
        info += "None (Scalar only)";
    }
    
    info += "\nSIMD Width: " + std::to_string(SIMD_WIDTH) + " floats";
    
    return info;
}

bool SIMDUtils::IsSIMDSupported() {
    #if defined(HAS_AVX512) || defined(HAS_AVX) || defined(HAS_SSE) || defined(HAS_NEON)
    return true;
    #else
    return false;
    #endif
}

int SIMDUtils::GetOptimalAlignment() {
    #ifdef HAS_AVX512
    return 64;  // 512-bit alignment
    #elif defined(HAS_AVX)
    return 32;  // 256-bit alignment
    #elif defined(HAS_SSE) || defined(HAS_NEON)
    return 16;  // 128-bit alignment  
    #else
    return 4;   // 32-bit alignment
    #endif
}

} // namespace Utils
} // namespace crllwtt