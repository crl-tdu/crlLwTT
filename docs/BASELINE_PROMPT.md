# crlNexus 開発ベースライン（STA統合版）

## 📋 プロジェクト概要
**crlNexus**: 自由エネルギー原理に基づくGRUベースの身体性エージェント群制御システム + **STORM汎用予測フレームワーク** + **STA適応的ヒューマン・インタフェース**
- **GitHubリポジトリ**: `https://github.com/crl-tdu/crlNexus.git`
- **ローカルパス**: `/Users/igarashi/local/project_workspace/crlNexus`
- **目標**: 内部予測モデルを，感度推定を統合した **「空気を読む」適応的ヒューマン・インタフェース（STA）**
- **革新的特徴**: **STAアーキテクチャ（LwTT統合）による人間中心の協調制御システム**
- **開発体制**: Header-onlyライブラリ設計, Doxygen形式の日本語コメント、Chain-of-Thought開発手法, CLionの使用
- **AI支援開発**: GitHubリポジトリからのコード参照・実装提案、テスト実装/テストビルドなどは必ず tmp/ 以下で実施（**プロジェクツリー内に許可なくファイルを作成しない**）、文書同期更新

```
ディスカッションにおいて，ステップバック方式として，議論を深めたい。コード作成を行う場合は，必ず許可をとること。
```

## 🔧 開発・協働方針

### Gitワークフロー
- **メインブランチ**: `main` - 安定版リリース用
- **開発ブランチ**: `develop` - 機能統合・テスト用
- **フィーチャーブランチ**: `feature/module-name` - 新機能開発用
- **コミットメッセージ**: `[モジュール名] 機能概要 - 詳細説明`
- **プルリクエスト**: コード更新時は必ず確認・許可を取ること

### コード開発原則
- **Header-only library**: 新規コードは原則ヘッダーオンリー（*.hpp）
- **統合ヘッダー**: 各モジュールに統合ヘッダー（*.h）作成
- **Doxygen形式**: `///` による日本語コメント（詳細・わかりやすく）
- **Chain-of-Thought**: ステップバイステップ開発・バックステップ形式（コード更新前に必ず確認）
- **テスト配置**: 必ず `tmp/` ディレクトリに配置

### AI開発支援プロトコル
1. **コード参照**: GitHubリポジトリから最新状況を確認
2. **実装提案**: 既存アーキテクチャとの整合性を重視
3. **テスト先行**: 新機能は必ずテストコード作成後に実装
4. **文書同期**: 実装完了後、該当ドキュメントを自動更新

### 文書作成方針
- **AI支援文書作成**: 自然言語での直接依頼による高速文書生成
- **リアルタイム対話**: 要求明確化→即座調査→反復改善による品質向上
- **即座反映**: 承認後の直接的なファイル作成・更新
- **柔軟性重視**: 途中変更・実験的文書・緊急対応に最適化
- **個人最適化**: Hiroshi教授の研究スタイル・専門分野に特化した支援

## 📁 システム構成

### ✅ **完全実装済みモジュール**
```
include/crlnexus/
├── swarm/                   # 🚀 群制御システム（完全実装・最適化済み）
│   ├── agents/             # GRUエージェント実装
│   │   ├── gru_agent.hpp   # GRU-FEP統合エージェント
│   │   └── polar_attention.hpp # 極座標空間注意機構
│   ├── managers/           # 管理システム（MessagePack統合）
│   │   ├── base_swarm_manager.hpp # 基底群管理クラス
│   │   └── swarm_coordinator.hpp  # 協調制御システム
│   ├── maps/              # 空間認識システム
│   │   └── spatial_map.hpp # 空間マップ管理
│   └── swarm.h            # 統合ヘッダー
├── tracking/               # 🎯 トラッキングシステム（BaseObject継承）
│   ├── core/              # TrackingObject、SecondOrderDynamics
│   │   ├── tracking_object.hpp    # 追従対象管理
│   │   └── second_order_dynamics.hpp # 2次系物理制御
│   └── tracking.h         # 統合ヘッダー
├── object/                 # 🏗️ 基本オブジェクト・設定システム
│   ├── core/              # BaseObject基底クラス
│   │   ├── base_object.hpp # 基底オブジェクト
│   │   └── config_manager.hpp # 設定管理システム
│   └── object.h           # 統合ヘッダー
├── utils/serialization/    # 🆕 MessagePack統合シリアライゼーション
│   ├── msgpack_handler.hpp # MessagePack処理
│   ├── json_fallback.hpp   # JSONフォールバック
│   └── serialization.h     # 統合ヘッダー
```

### 🌟 **STA（Sense The Ambience）アーキテクチャ（新規実装完了）**
```
include/sta/                # 🔥 STA適応的ヒューマン・インタフェース
├── core/                   # 核心コンポーネント
│   ├── lwtt_prediction_module.hpp  # LwTT統合予測モジュール
│   ├── sensitivity_module.hpp      # 感度分析モジュール
│   └── control_module.hpp          # 適応的制御モジュール
├── interfaces/             # 抽象インターフェース
│   ├── sensor_interface.hpp        # センサー抽象化
│   ├── actuator_interface.hpp      # アクチュエータ抽象化
│   └── objective_interface.hpp     # 目的関数抽象化
├── applications/           # 応用分野（実装予定）
│   ├── tracking/           # トラッキング統合
│   ├── automotive/         # 自動運転支援
│   └── medical/            # 医療支援
└── sta.h                   # 統合ヘッダー
```

### 🌟 **STORM汎用予測フレームワーク（実装完了）**
```
include/storm/              # 🔥 STORM汎用予測フレームワーク
├── core/                   # 核心コンポーネント（学習器非依存）
│   ├── personality_vector.hpp  # 6次元個性ベクトル（単体制約）
│   ├── som_meta_evaluator.hpp # SOMメタ評価・構造化知識管理
│   ├── ensemble_manager.hpp    # 多様性制御GRUアンサンブル
│   └── imitation_learning.hpp  # 動的知識伝播・相互模倣学習
├── predictors/             # 学習器実装
│   ├── predictor_interface.hpp # 学習器非依存インターフェース
│   ├── gru_predictor.hpp       # STORM-GRU実装
│   └── extended_gru.hpp        # 個性ベクトル拡張GRU
├── applications/           # 応用分野（今後実装）
│   ├── tracking/           # トラッキング統合（人間操作予測）
│   ├── financial/          # 金融予測（概念ドリフト対応）
│   └── manufacturing/      # 製造品質予測
└── storm.h                 # 統合ヘッダー

external/crlLwTT/          # 🔥 軽量時間認識Transformerライブラリ（参照）
├── include/LwTT/          # LwTTコアライブラリ
│   ├── LwTT.hpp          # メインヘッダー
│   ├── core/             # 核心コンポーネント
│   │   ├── STATransformer.hpp    # STA（Sense The Ambience）
│   │   ├── Transformer.hpp      # 時間認識Transformer
│   │   ├── SparseAttention.hpp  # スパース注意機構
│   │   ├── TimeEncoding.hpp     # 時間エンコーディング
│   │   └── Tensor.hpp           # テンソル操作
│   ├── layers/           # Transformerレイヤー
│   └── utils/            # ユーティリティ
└── tests/examples/       # 使用例・ベンチマーク
```

### 📊 **現在の状況（2025年6月2日・STA完全実装完了）**
- **全テストスイート**: ✅ 完全動作確認済み
- **swarm_test.cpp**: 平均ループ時間24.54ms、移動成功率100%
- **gruagent_test.cpp**: FEP-GRU統合（20エージェント）、PolarSpatialAttention動作
- **solo_tracking_test.cpp**: 2次系物理制御、円軌道追従（平均誤差2.67）
- **MessagePack統合**: JSON/MessagePack動的切り替え、フォールバック機能
- **🌟 STA完全実装**: LwTT統合による「空気を読む」適応的ヒューマン・インタフェース
- **🔥 知覚-予測-行動サイクル**: リアルタイム勾配ベース制御（180Hz動作可能）
- **🌟 STORM理論設計完了**: 個性ベクトル・SOMメタ評価・相互模倣学習の統合フレームワーク
- **📈 期待性能**: STORM予測精度40%改善、適応速度50%向上、制御安定性99.8%、総サイクル時間5.5ms

### 🔍 **Git状況監視指標**
- **最新コミット**: `git log --oneline -5` で確認
- **ブランチ状況**: `git branch -a` で開発状況把握
- **未追跡ファイル**: `git status` で作業状況確認
- **差分確認**: `git diff HEAD~1` で最新変更把握

## 🔬 理論的枠組み

### STA（Sense The Ambience）数学的定式化
#### 知覚-予測-行動サイクル
```
1. 状態予測（LwTT統合）:
   ŝ[k] = LwTT_STA(x[k], u[k], TimeInfo)

2. 目的関数評価:
   M(ŝ) = サプライズ関数（望ましくない状態の定量化）

3. 感度分析（連鎖律）:
   ∇_u M(ŝ) = ∂M/∂ŝ · ∂ŝ/∂u

4. 適応的制御更新:
   u*[k+1] = u[k] - α[k] · ∇_u M(ŝ)
   α[k] = 適応的学習率（勾配ノルム・信頼度ベース）
```

### 変分自由エネルギー最小化
$$F_t^{(i)} = D_{KL}[q_{\phi^{(i)}}(x_t^{(i)}) || p(x_t^{(i)} | y_t^{(i)})] + H[p(y_t^{(i)})]$$

### STORM理論統合（学習器非依存）

#### STORM（Self-Organizing-Map-guided Temporal Orchestrated Recurrent Model）概要
STORMは人間機械協調システムにおける人間操作予測の精度向上とコンセプトドリフト適応を同時に解決する革新的アーキテクチャです。**個性ベクトル**、**SOMメタ評価**、**相互模倣学習**の3つの核心技術を統合します。

#### 6次元個性ベクトル（単体制約）
各GRU i は以下の6次元個性ベクトル θ^(i) により特性化されます：

```
θ^(i) = [θ₁^(i), θ₂^(i), θ₃^(i), θ₄^(i), θ₅^(i), θ₆^(i)]ᵀ

θ₁: 記憶保持度 (Memory Persistence)
θ₂: 変化感度 (Change Sensitivity)  
θ₃: 学習機敏性 (Learning Agility)
θ₄: ノイズ頑健性 (Noise Robustness)
θ₅: 時間焦点 (Temporal Focus)
θ₆: 探索傾向 (Exploration Tendency)

制約: Σⱼ₌₁⁶ θⱼ^(i) = 1, θⱼ^(i) ≥ 0
```

#### 個性駆動アンサンブル予測（汎用）
$$\hat{y}^{(i)}[k] = \text{Predictor}_{\theta^{(i)}}(\mathbf{x}[k], \mathbf{h}_{t-1}^{(i)})$$
$$\text{制約}: \sum_{j=1}^{6} \theta_j^{(i)} = 1, \theta_j^{(i)} \geq 0$$

ここで$\text{Predictor}_{\theta^{(i)}}$は以下の学習器に対応：
- **LwTT**: $\text{LwTT}_{\theta^{(i)}}(\mathbf{x}[k], \text{TimeInfo}_{\theta^{(i)}})$
- **GRU**: $\text{GRU}_{\theta^{(i)}}(\mathbf{x}[k], \mathbf{h}_{t-1}^{(i)})$
- **Transformer**: $\text{Transformer}_{\theta^{(i)}}(\mathbf{x}[k], \text{Attention}_{\theta^{(i)}})$

#### SOMメタ評価システム（構造化知識管理）
**最良マッチングユニット（BMU）**：
$$\text{BMU} = \arg\min_{(m,n)} ||\mathbf{h}^{(i)}[k] - \mathbf{w}_{m,n}||^2$$

**多層残差マップ**：
- 即時残差：$E_{i,m,n}^{\text{inst}}[k] = L^{(i)}[k]$
- 指数移動平均残差：$E_{i,m,n}^{\text{ema}}[k+1] = (1-\alpha_{ema}) E_{i,m,n}^{\text{ema}}[k] + \alpha_{ema} L^{(i)}[k]$
- 信頼性指標：$R_{i,m,n}[k] = \frac{1}{\text{Var}[L^{(i)}] + \epsilon}$

#### 相互模倣学習（Knowledge Propagation）
**階層的カリスマ選択**：
$$j^*_{m,n}[k] = \arg\max_i S_{i,m,n}[k]$$

**模倣強度計算**：
$$I_{i \rightarrow j^*} = \exp(-\gamma \frac{L^{(i)}}{L^{(j^*)}}) \cdot \text{sim}(\theta^{(i)}, \theta^{(j^*)})$$

**構造化模倣更新**：
$$\theta^{(i)}[k+1] = \theta^{(i)}[k] + \alpha[k] \cdot I_{i \rightarrow j^*} \cdot (\theta^{(j^*)} - \theta^{(i)}) + \sigma[k]\xi$$

**適応的制御パラメータ**：
- 残差エントロピー：$H_L[k] = -\sum_{i=1}^{G} \pi^{(i)}[k] \log \pi^{(i)}[k]$
- 模倣率：$\alpha[k] = \alpha_{max} \cdot \tanh(\beta \cdot \frac{H_L[k]}{\log G})$

#### サプライズ推定（自由エネルギー原理）
$$S^{(i)}[k] = ||\hat{y}^{(i)}[k] - y[k]||^2 \cdot (1 + \theta_2^{(i)} + \theta_4^{(i)})$$
$$S_{\text{STORM}}[k] = \frac{\sum_i w_i[k] \cdot S^{(i)}[k]}{\sum_i w_i[k]}$$

#### 理論的保証
- **収束性**: 予測誤差の指数収束 $L_{max}[k+1] \leq (1 - \alpha_{min}[k])L_{max}[k]$
- **多様性保持**: エントロピー下限 $H_\theta[k] \geq H_{min} > 0$
- **確率的安定性**: リアプノフ関数による安定性保証

### 極座標空間注意機構
$$\text{Attention}(\mathbf{X}_{polar}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$
$$\mathbf{F}_{attention} \in \mathbb{R}^{108}, \quad \mathbb{E}[\mathbf{F}_{attention}] = -0.27 \pm 1.37$$

### 2次系トラッキング制御
$$\mathbf{M} \ddot{\mathbf{x}} + \mathbf{B} \dot{\mathbf{x}} = \mathbf{G} \mathbf{u}$$
$$\mathbf{u} = K_p (\mathbf{x}_{ref} - \mathbf{x}) + K_d (\dot{\mathbf{x}}_{ref} - \dot{\mathbf{x}})$$

### MessagePack通信効率化
$$F_{comm} = H[p(data)] - D_{KL}[q(compressed) || p(original)]$$
$$\text{Compression Efficiency} = \frac{\text{MessagePack Size}}{\text{JSON Size}} \approx 0.75$$

## 🔧 開発環境・ツールチェーン

### 基本環境
- **LibTorch**: `/Users/igarashi/local/libtorch` (2.1.2)
- **ビルド**: `debug/` でテスト実行
- **可視化**: `python/visualizer/` (object_viewer.py, udp_dual_monitor.py)
- **コンパイラ**: AppleClang 17.0.0 (macOS ARM64)

### Git統合開発コマンド
```bash
# 開発開始時の準備
git fetch origin && git status
git checkout develop
git pull origin develop

# フィーチャーブランチ作成
git checkout -b feature/new-module-name

# 開発完了時のマージ準備
git add . && git commit -m "[モジュール名] 機能実装完了"
git push origin feature/new-module-name

# テスト実行（ビルド前にGit状況確認）
cd debug && make && ./target_test
```

### AI支援開発フロー
1. **現状分析**: `git status` + リポジトリ参照で開発状況把握
2. **要件定義**: Chain-of-Thoughtで段階的実装計画策定
3. **実装支援**: 既存コードベースとの整合性確保
4. **テスト生成**: 新機能に対応するテストコード自動生成
5. **文書更新**: README、docs/の関連文書自動更新

### Pattern B文書作成フロー（採用済み）
```
文書作成依頼 → AI：要件分析・質問 → 詳細要求明確化 → AI：リポジトリ調査
→ AI：文書構成提案 → 承認・修正指示 → AI：文書生成・コード例作成
→ リアルタイム修正・改善 → 最終確認 → リポジトリ直接反映
```

**依頼例**:
- 「STAモジュールのAPI文書を作成してください」
- 「STA-LwTT統合の実装ガイドが必要です」
- 「今日の実験結果を分析レポートにまとめてください」

## 🎯 検証対象テスト
1. `tests/swarm/gruagent_test.cpp` - FEP-GRU統合テスト
2. `tests/swarm/swarm_test.cpp` - BaseSwarmManager統合テスト  
3. `tests/tracking/solo/solo_tracking_test.cpp` - TrackingObject物理制御テスト
4. `tests/object/object_test.cpp` - BaseObject基本機能テスト
5. **🌟 NEW** `tests/storm_lwtt/storm_lwtt_test.cpp` - STORM-LwTT統合予測システムテスト
6. **🔥 NEW** `tests/sta/sta_system_test.cpp` - STA完全統合システムテスト

## 🎯 今後の開発ロードマップ

### 🚀 短期目標（今後2週間）- STA-Tracking統合
1. **STATrackingAgent実装**: STAアーキテクチャとcrlNexusの統合
2. **実証実験システム**: トラッキングタスクでの性能検証
3. **性能最適化**: リアルタイム推論（5.5ms→3.0ms）達成

### 🔥 中期目標（今後1ヶ月）- 応用展開
4. **自動運転支援応用**: 操舵微調整・安全性向上システム
5. **医療支援応用**: リハビリテーション・手術支援システム
6. **教育システム応用**: 適応的学習支援インターフェース

### 🌟 長期目標（今後3ヶ月）- 社会実装
7. **産業連携**: 実用化パートナーとの協業開始
8. **国際会議発表**: STA論文の査読付き会議投稿・発表
9. **オープンソース展開**: GitHubでの公開・コミュニティ形成

## 📚 開発リソース・文書体系

### コア文書（リポジトリ内）
- 📘 `docs/BASELINE_COMPACT.md` - 完全開発仕様書（本文書）
- 📋 `docs/CHANGELOG.md` - 版数管理・更新履歴
- 🏗️ `docs/DIRECTORY_STRUCTURE.md` - アーキテクチャ詳細
- 🔬 `docs/RESEARCH_THEORY.md` - 理論的背景・数学的定式化
- **🌟 NEW** `docs/STA_TECHNICAL_NOTE.md` - STAアーキテクチャ技術文書

### 実装文書
- 🤖 `docs/GRU_AGENT_MANUAL.md` - GRUエージェント実装ガイド
- 📡 `docs/UDP_PROTOCOL_SPEC.md` - 通信プロトコル仕様
- 🧪 `docs/TEST_SPECIFICATIONS.md` - テスト仕様・検証手順
- **🔥 NEW** `docs/STA_IMPLEMENTATION_GUIDE.md` - STA実装・使用ガイド

### Git連携文書
- 🔄 `docs/GIT_WORKFLOW.md` - ブランチ戦略・協働指針
- 🤖 `docs/AI_DEVELOPMENT_GUIDE.md` - AI支援開発プロトコル

## 🌟 STA使用クイックスタート

```cpp
#include "sta/sta.h"

int main() {
    // STAシステム初期化
    auto sta_system = std::make_shared<sta::STASystem>(8, 4, 6);
    
    // 目的関数・センサー・アクチュエータ登録
    sta_system->registerObjective("tracking_error", tracking_objective);
    sta_system->registerSensor("heart_rate", hr_sensor);
    sta_system->registerActuator("display_brightness", display_actuator);
    
    // リアルタイム「空気を読む」制御
    for (int i = 0; i < 1000; ++i) {
        auto stats = sta_system->executeCycle();
        std::cout << "Success Rate: " << stats.getSuccessRate() 
                  << ", Cycle Time: " << stats.getAverageCycleTime() << "ms" << std::endl;
    }
    
    return 0;
}
```

---

**🚀 開発加速化のためのクイックアクセス**:
- **リポジトリ**: https://github.com/crl-tdu/crlNexus.git
- **課題追跡**: GitHubのIssuesで機能要求・バグ報告管理
- **CI/CD**: GitHub Actionsで自動テスト・ビルド検証
- **コードレビュー**: プルリクエストでの品質確保
- **🌟 STA文書**: `docs/STA_TECHNICAL_NOTE.md` で詳細技術仕様を確認