/**
 * @file SIMD.hpp
 * @brief SIMD optimized operations for high-performance computing
 * @version 1.0.0
 * @date 2025-05-25
 */

#ifndef LWTT_UTILS_SIMD_HPP
#define LWTT_UTILS_SIMD_HPP

#include <string>
#include <cmath>

namespace crllwtt {
namespace Utils {

/**
 * @brief Activation function types for SIMD optimization
 */
enum class ActivationType {
    ReLU,      ///< Rectified Linear Unit
    Tanh,      ///< Hyperbolic tangent
    Sigmoid,   ///< Sigmoid function
    GELU       ///< Gaussian Error Linear Unit
};

/**
 * @brief SIMD optimized utility functions
 */
class SIMDUtils {
public:
    /// SIMD vector width (number of floats)
    static const int kSIMDWidth;

    /**
     * @brief Vector addition: result = a + b
     * @param a First input vector
     * @param b Second input vector
     * @param result Output vector
     * @param size Number of elements
     */
    static void VectorAdd(const float* a, const float* b, float* result, int size);

    /**
     * @brief Element-wise vector multiplication: result = a * b
     * @param a First input vector
     * @param b Second input vector
     * @param result Output vector
     * @param size Number of elements
     */
    static void VectorMul(const float* a, const float* b, float* result, int size);

    /**
     * @brief Dot product: result = sum(a[i] * b[i])
     * @param a First input vector
     * @param b Second input vector
     * @param size Number of elements
     * @return Dot product result
     */
    static float DotProduct(const float* a, const float* b, int size);

    /**
     * @brief Matrix multiplication: C = A * B
     * @param a Matrix A (m x k)
     * @param b Matrix B (k x n)
     * @param c Output matrix C (m x n)
     * @param m Number of rows in A
     * @param n Number of columns in B
     * @param k Common dimension
     * @param transpose_a Whether to transpose matrix A
     * @param transpose_b Whether to transpose matrix B
     */
    static void MatrixMultiply(const float* a, const float* b, float* c,
                              int m, int n, int k,
                              bool transpose_a = false, bool transpose_b = false);

    /**
     * @brief Apply activation function with SIMD optimization
     * @param input Input vector
     * @param output Output vector
     * @param size Number of elements
     * @param activation Type of activation function
     */
    static void ApplyActivation(const float* input, float* output, int size, ActivationType activation);

    /**
     * @brief Softmax function with SIMD optimization
     * @param input Input vector
     * @param output Output vector (normalized probabilities)
     * @param size Number of elements
     */
    static void SoftMax(const float* input, float* output, int size);

    /**
     * @brief Vector scaling: result = input * scale
     * @param input Input vector
     * @param scale Scalar multiplier
     * @param output Output vector
     * @param size Number of elements
     */
    static void VectorScale(const float* input, float scale, float* output, int size);

    /**
     * @brief Get SIMD capability information
     * @return String describing available SIMD instructions
     */
    static std::string GetSIMDInfo();

    /**
     * @brief Check if SIMD is supported on current platform
     * @return True if SIMD is available
     */
    static bool IsSIMDSupported();

    /**
     * @brief Get optimal memory alignment for SIMD operations
     * @return Alignment in bytes
     */
    static int GetOptimalAlignment();

private:
    SIMDUtils() = delete; // Static class only
};

/**
 * @brief SIMD-optimized memory allocation helper
 */
class SIMDAllocator {
public:
    /**
     * @brief Allocate aligned memory for SIMD operations
     * @param size Number of floats to allocate
     * @return Aligned memory pointer, or nullptr on failure
     */
    static float* AllocateAligned(size_t size);

    /**
     * @brief Free aligned memory
     * @param ptr Pointer returned by AllocateAligned
     */
    static void FreeAligned(float* ptr);

    /**
     * @brief Check if pointer is properly aligned for SIMD
     * @param ptr Pointer to check
     * @return True if aligned
     */
    static bool IsAligned(const void* ptr);
};

} // namespace Utils
} // namespace crllwtt

#endif // LWTT_UTILS_SIMD_HPP