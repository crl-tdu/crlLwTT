#include <gtest/gtest.h>
#include <LwTT/core/Tensor.hpp>

namespace {

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {
        // テスト前の設定
    }
};

TEST_F(TensorTest, Initialization) {
    // ベクトル
    EXPECT_NO_THROW({
        crllwtt::Core::Tensor vector({10});
        EXPECT_EQ(vector.GetShape().size(), 1);
        EXPECT_EQ(vector.GetShape()[0], 10);
        EXPECT_EQ(vector.GetSize(), 10);
    });
    
    // 行列
    EXPECT_NO_THROW({
        crllwtt::Core::Tensor matrix({5, 10});
        EXPECT_EQ(matrix.GetShape().size(), 2);
        EXPECT_EQ(matrix.GetShape()[0], 5);
        EXPECT_EQ(matrix.GetShape()[1], 10);
        EXPECT_EQ(matrix.GetSize(), 50);
    });
    
    // 3次元テンソル
    EXPECT_NO_THROW({
        crllwtt::Core::Tensor tensor({2, 3, 4});
        EXPECT_EQ(tensor.GetShape().size(), 3);
        EXPECT_EQ(tensor.GetShape()[0], 2);
        EXPECT_EQ(tensor.GetShape()[1], 3);
        EXPECT_EQ(tensor.GetShape()[2], 4);
        EXPECT_EQ(tensor.GetSize(), 24);
    });
}

TEST_F(TensorTest, DataAccess) {
    crllwtt::Core::Tensor tensor({2, 3});
    
    // 値の設定と取得
    tensor.SetValue({0, 0}, 1.0f);
    tensor.SetValue({0, 1}, 2.0f);
    tensor.SetValue({0, 2}, 3.0f);
    tensor.SetValue({1, 0}, 4.0f);
    tensor.SetValue({1, 1}, 5.0f);
    tensor.SetValue({1, 2}, 6.0f);
    
    EXPECT_FLOAT_EQ(tensor.GetValue({0, 0}), 1.0f);
    EXPECT_FLOAT_EQ(tensor.GetValue({0, 1}), 2.0f);
    EXPECT_FLOAT_EQ(tensor.GetValue({0, 2}), 3.0f);
    EXPECT_FLOAT_EQ(tensor.GetValue({1, 0}), 4.0f);
    EXPECT_FLOAT_EQ(tensor.GetValue({1, 1}), 5.0f);
    EXPECT_FLOAT_EQ(tensor.GetValue({1, 2}), 6.0f);
    
    // 範囲外アクセス
    EXPECT_THROW(tensor.SetValue({2, 0}, 7.0f), std::out_of_range);
    EXPECT_THROW(tensor.SetValue({0, 3}, 8.0f), std::out_of_range);
    EXPECT_THROW(tensor.GetValue({2, 0}), std::out_of_range);
    EXPECT_THROW(tensor.GetValue({0, 3}), std::out_of_range);
}

TEST_F(TensorTest, Operations) {
    crllwtt::Core::Tensor a({2, 2});
    crllwtt::Core::Tensor b({2, 2});
    
    // 値を設定
    a.SetValue({0, 0}, 1.0f);
    a.SetValue({0, 1}, 2.0f);
    a.SetValue({1, 0}, 3.0f);
    a.SetValue({1, 1}, 4.0f);
    
    b.SetValue({0, 0}, 5.0f);
    b.SetValue({0, 1}, 6.0f);
    b.SetValue({1, 0}, 7.0f);
    b.SetValue({1, 1}, 8.0f);
    
    // 加算
    EXPECT_NO_THROW({
        crllwtt::Core::Tensor c = a.Add(b);
        EXPECT_FLOAT_EQ(c.GetValue({0, 0}), 6.0f);
        EXPECT_FLOAT_EQ(c.GetValue({0, 1}), 8.0f);
        EXPECT_FLOAT_EQ(c.GetValue({1, 0}), 10.0f);
        EXPECT_FLOAT_EQ(c.GetValue({1, 1}), 12.0f);
    });
    
    // 乗算
    EXPECT_NO_THROW({
        crllwtt::Core::Tensor d = a.Multiply(b);
        EXPECT_FLOAT_EQ(d.GetValue({0, 0}), 5.0f);
        EXPECT_FLOAT_EQ(d.GetValue({0, 1}), 12.0f);
        EXPECT_FLOAT_EQ(d.GetValue({1, 0}), 21.0f);
        EXPECT_FLOAT_EQ(d.GetValue({1, 1}), 32.0f);
    });
    
    // スカラー乗算
    EXPECT_NO_THROW({
        crllwtt::Core::Tensor e = a.MultiplyScalar(2.0f);
        EXPECT_FLOAT_EQ(e.GetValue({0, 0}), 2.0f);
        EXPECT_FLOAT_EQ(e.GetValue({0, 1}), 4.0f);
        EXPECT_FLOAT_EQ(e.GetValue({1, 0}), 6.0f);
        EXPECT_FLOAT_EQ(e.GetValue({1, 1}), 8.0f);
    });
}

} // namespace
