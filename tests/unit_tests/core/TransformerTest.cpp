#include <gtest/gtest.h>
#include <LwTT/core/Transformer.hpp>
#include <LwTT/core/TimeEncoding.hpp>

namespace {

class TransformerTest : public ::testing::Test {
protected:
    void SetUp() override {
        // テスト前の設定
        config.d_model = 64;
        config.n_heads = 4;
        config.n_layers = 2;
        config.max_seq_len = 100;
        config.enable_time_encoding = true;
        config.use_sparse_attention = false;
    }

    crllwtt::Core::TransformerConfig config;
};

TEST_F(TransformerTest, Initialization) {
    ASSERT_NO_THROW({
        auto transformer = std::make_unique<crllwtt::Core::Transformer>(config);
    });
}

TEST_F(TransformerTest, ForwardPass) {
    auto transformer = std::make_unique<crllwtt::Core::Transformer>(config);
    
    // 入力テンソルの作成
    crllwtt::Core::Tensor input({1, 10, config.d_model});
    
    // 時間情報の設定
    crllwtt::Core::TimeInfo time_info;
    for (int i = 0; i < 10; ++i) {
        time_info.timestamps.push_back(i * 0.1f);
    }
    time_info.ComputeDeltas();
    
    // 順伝播処理
    ASSERT_NO_THROW({
        auto output = transformer->Forward(input, nullptr, &time_info);
        EXPECT_EQ(output.GetShape()[0], 1);
        EXPECT_EQ(output.GetShape()[1], 10);
        EXPECT_EQ(output.GetShape()[2], config.d_model);
    });
}

} // namespace
