#include <gtest/gtest.h>
#include <LwTT/core/TimeEncoding.hpp>
#include <vector>

namespace {

class TimeEncodingTest : public ::testing::Test {
protected:
    void SetUp() override {
        // テスト前の設定
        config.d_model = 128;
        config.max_seq_len = 200;
        config.time_scale = 1.0f;
        config.enable_time_encoding = true;
        config.enable_time_scaling = true;
        config.personal_embed_dim = 32;
    }

    LwTT::Core::TimeEncodingConfig config;
};

TEST_F(TimeEncodingTest, Initialization) {
    ASSERT_NO_THROW({
        auto time_encoding = std::make_unique<LwTT::Core::TimeEncoding>(config);
    });
}

TEST_F(TimeEncodingTest, TimeInfoCreation) {
    // タイムスタンプの作成
    std::vector<float> timestamps;
    for (int i = 0; i < 10; ++i) {
        timestamps.push_back(i * 0.1f);
    }
    
    // TimeInfoの作成
    auto time_info = LwTT::Core::TimeEncodingUtils::CreateTimeInfo(timestamps, 0.05f);
    
    // 検証
    EXPECT_EQ(time_info.timestamps.size(), 10);
    EXPECT_EQ(time_info.time_deltas.size(), 9);
    EXPECT_FLOAT_EQ(time_info.personal_delay, 0.05f);
    
    // 時間差分の検証
    for (size_t i = 0; i < time_info.time_deltas.size(); ++i) {
        EXPECT_FLOAT_EQ(time_info.time_deltas[i], 0.1f);
    }
}

TEST_F(TimeEncodingTest, EncodingApplication) {
    auto time_encoding = std::make_unique<LwTT::Core::TimeEncoding>(config);
    
    // 入力テンソルの作成
    LwTT::Core::Tensor input({1, 10, 64});
    
    // TimeInfoの作成
    std::vector<float> timestamps;
    for (int i = 0; i < 10; ++i) {
        timestamps.push_back(i * 0.1f);
    }
    auto time_info = LwTT::Core::TimeEncodingUtils::CreateTimeInfo(timestamps);
    
    // エンコーディングの適用
    ASSERT_NO_THROW({
        auto encoded = time_encoding->Apply(input, &time_info);
        EXPECT_EQ(encoded.GetShape()[0], 1);
        EXPECT_EQ(encoded.GetShape()[1], 10);
        EXPECT_EQ(encoded.GetShape()[2], 64);  // 入力次元数と同じであることを確認
    });
}

} // namespace
