/**
 * @file Tensor.hpp
 * @brief Header for Tensor class
 * @version 1.0.0
 * @date 2025-05-25
 */

#ifndef LWTT_CORE_TENSOR_HPP
#define LWTT_CORE_TENSOR_HPP

#include <vector>
#include <string>

namespace LwTT {
namespace Core {

/**
 * @brief Tensor class for n-dimensional array operations
 */
class Tensor {
public:
    /**
     * @brief Default constructor (creates empty tensor)
     */
    Tensor();
    
    /**
     * @brief Constructor
     * @param shape Shape of the tensor
     */
    explicit Tensor(const std::vector<int>& shape);
    
    /**
     * @brief Copy constructor
     * @param other Other tensor to copy from
     */
    Tensor(const Tensor& other);
    
    /**
     * @brief Move constructor
     * @param other Other tensor to move from
     */
    Tensor(Tensor&& other) noexcept;
    
    /**
     * @brief Copy assignment operator
     * @param other Other tensor to copy from
     * @return Reference to this tensor
     */
    Tensor& operator=(const Tensor& other);
    
    /**
     * @brief Move assignment operator
     * @param other Other tensor to move from
     * @return Reference to this tensor
     */
    Tensor& operator=(Tensor&& other) noexcept;
    
    /**
     * @brief Destructor
     */
    ~Tensor();
    
    /**
     * @brief Get value at specified indices
     * @param indices Indices in each dimension
     * @return Value at the specified position
     */
    float GetValue(const std::vector<int>& indices) const;
    
    /**
     * @brief Set value at specified indices
     * @param indices Indices in each dimension
     * @param value Value to set
     */
    void SetValue(const std::vector<int>& indices, float value);
    
    /**
     * @brief Get raw data pointer
     * @return Pointer to the data
     */
    float* GetData() { return data_; }
    
    /**
     * @brief Get raw data pointer (const)
     * @return Const pointer to the data
     */
    const float* GetData() const { return data_; }
    
    /**
     * @brief Get tensor shape
     * @return Vector of dimensions
     */
    const std::vector<int>& GetShape() const { return shape_; }
    
    /**
     * @brief Get total number of elements
     * @return Size of the tensor
     */
    int GetSize() const { return size_; }
    
    /**
     * @brief Add another tensor element-wise
     * @param other Other tensor to add
     * @return Result tensor
     */
    Tensor Add(const Tensor& other) const;
    
    /**
     * @brief Multiply another tensor element-wise
     * @param other Other tensor to multiply
     * @return Result tensor
     */
    Tensor Multiply(const Tensor& other) const;
    
    /**
     * @brief Multiply by scalar
     * @param scalar Scalar value
     * @return Result tensor
     */
    Tensor MultiplyScalar(float scalar) const;
    
    /**
     * @brief Get value at specified indices (alias for GetValue)
     * @param indices Indices in each dimension
     * @return Value at the specified position
     */
    float Get(const std::vector<int>& indices) const { return GetValue(indices); }
    
    /**
     * @brief Set value at specified indices (alias for SetValue)
     * @param indices Indices in each dimension
     * @param value Value to set
     */
    void Set(const std::vector<int>& indices, float value) { SetValue(indices, value); }
    
    /**
     * @brief Get tensor shape (alias for GetShape)
     * @return Vector of dimensions
     */
    const std::vector<int>& Shape() const { return GetShape(); }
    
    /**
     * @brief Fill tensor with specified value
     * @param value Value to fill with
     */
    void Fill(float value);
    
    /**
     * @brief Fill tensor with random normal values
     * @param mean Mean of the distribution
     * @param std Standard deviation of the distribution
     */
    void RandomNormal(float mean, float std);
    
    /**
     * @brief Multiply tensor by scalar in-place
     * @param scalar Scalar value
     */
    void Multiply(float scalar);
    
    /**
     * @brief Get memory size in bytes
     * @return Memory size
     */
    size_t GetMemorySize() const { return size_ * sizeof(float); }
    
    /**
     * @brief Create a slice of the tensor
     * @param dim Dimension to slice
     * @param start Start index
     * @param end End index
     * @return Sliced tensor
     */
    Tensor Slice(int dim, int start, int end) const;
    
    /**
     * @brief Set a slice of the tensor
     * @param dim Dimension to slice
     * @param start Start index
     * @param end End index
     * @param values Values to set
     */
    void SetSlice(int dim, int start, int end, const Tensor& values);
    
    /**
     * @brief Get string representation of tensor shape
     * @return Shape as string
     */
    std::string ShapeString() const;
    
private:
    float* data_;                 ///< Raw data storage
    std::vector<int> shape_;      ///< Shape of the tensor
    int size_;                    ///< Total number of elements
    
    /**
     * @brief Convert n-dimensional indices to flat index
     * @param indices Indices in each dimension
     * @return Flat index
     */
    int FlattenIndices(const std::vector<int>& indices) const;
};

} // namespace Core
} // namespace LwTT

#endif // LWTT_CORE_TENSOR_HPP
