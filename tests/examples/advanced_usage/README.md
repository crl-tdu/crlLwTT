# STA Architecture Integration Example

This example demonstrates the integration of the **Sense The Ambience (STA) Architecture** with the **Lightweight Time-aware Transformer (LwTT)** library.

## Overview

The STA Architecture enables real-time adaptive control of human-computer interaction systems by:

1. **Real-time State Prediction**: Predicting future human internal states (concentration, stress, fatigue, valence) based on observable indicators
2. **Sensitivity Analysis**: Computing how control inputs affect predicted states (∂ŝ/∂u)
3. **Adaptive Control**: Optimizing environmental inputs to achieve desired human states
4. **Online Learning**: Continuously adapting to individual user characteristics

## Key Features Demonstrated

### 1. Multi-modal Human State Modeling
- **Observable States (12D)**: Heart rate, skin conductance, blink frequency, gaze dispersion, posture, typing speed, mouse movement, EEG alpha/beta, head orientation, voice tone, breathing rate
- **Control Inputs (8D)**: Lighting brightness, color temperature, volume, notification frequency, room temperature, information presentation speed, background sound, airflow
- **Predicted States (4D)**: Concentration, stress, fatigue, emotional valence

### 2. Real-time Performance
- **Sub-5ms inference latency** for state prediction
- **Microsecond-level sensitivity computation** with caching
- **Ring buffer management** for continuous data streams
- **Online learning** with adaptive learning rates

### 3. Adaptive Control Strategies
- **Sensitivity-based optimization**: Uses ∂ŝ/∂u to determine optimal control adjustments
- **Meta-evaluation functions**: Customizable objective functions for different user goals
- **Multi-objective optimization**: Balance between concentration, stress reduction, and other factors
- **Personal adaptation**: Individual delay compensation and preference learning

## Running the Example

### Prerequisites
- C++17 compatible compiler
- CMake 3.16+
- LwTT library (built from source)

### Build and Run
```bash
# From the project root directory
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make sta_integration_example

# Run the example
./examples/sta_integration_example
```

## Technical Implementation Details

### STA Transformer Architecture
```cpp
auto sta_transformer = STATransformerBuilder()
    .SetModelDimension(128)           // Compact model for real-time performance
    .SetNumHeads(8)                   // Multi-head attention
    .SetNumLayers(4)                  // Balanced depth vs. speed
    .SetControlInputDim(8)            // Environmental control dimensions
    .SetObservableStateDim(12)        // Sensor input dimensions
    .SetPredictedStateDim(4)          // Internal state predictions
    .SetPredictionHorizon(10)         // Future prediction steps
    .SetMaxLatency(5.0f)              // Real-time constraint
    .EnableSensitivityCache(true)     // Performance optimization
    .EnableRingBuffer(true, 1000)     // Streaming data management
    .SetUncertaintyPenaltyWeight(0.1f) // Risk-aware control
    .Build();
```

### Core STA Loop
```cpp
// 1. Predict future human state
auto predicted_state = sta_transformer->PredictFutureState(
    observable_state, control_input, time_info, personal_id
);

// 2. Compute sensitivity ∂ŝ/∂u
auto sensitivity = sta_transformer->ComputeSensitivity(
    observable_state, control_input, time_info, personal_id
);

// 3. Optimize control inputs
auto optimal_control = sta_transformer->ComputeOptimalControl(
    observable_state, control_input, time_info, meta_evaluator, learning_rate, personal_id
);

// 4. Learn from observed outcomes
float loss = sta_transformer->OnlineLearning(
    observable_state, true_state, control_input, time_info, learning_rate, personal_id
);
```

## Research Integration

This example demonstrates the practical implementation of concepts from the research paper:
> "空気を読めるアーキテクチャ：任意入力に対するリアルタイム状態予測と感受性制御に基づく適応的ヒューマン・インタフェース"

The implementation showcases:

- **Enactive Cognition**: Active environment manipulation for learning and adaptation
- **Predictive Processing**: Future state estimation based on current observations and control inputs
- **Sensorimotor Contingency**: Learning the relationship between control actions and state changes
- **Free Energy Minimization**: Optimizing control to reduce prediction error and achieve target states

### Mathematical Foundation

The core STA equation implemented:

```
ŝ[k] = NN_θ(x[k-1], u[k-1])
u[k] = u[k-1] + η_u (∂ŝ[k]/∂u[k-1])^T ∇_ŝ[k] J(ŝ[k])
```

Where:
- `ŝ[k]`: Predicted future human state
- `x[k-1]`: Observable state indicators
- `u[k-1]`: Control inputs
- `J(ŝ[k])`: Meta-evaluation function
- `η_u`: Control learning rate

## Applications

This STA architecture can be applied to:

- **Smart Office Environments**: Optimizing lighting, temperature, and sound for productivity
- **Educational Systems**: Adapting content difficulty and presentation based on student state
- **Healthcare**: Monitoring and managing patient stress and comfort levels
- **Autonomous Vehicles**: Adjusting cabin environment based on driver alertness
- **Gaming and Entertainment**: Dynamic difficulty adjustment and experience optimization
- **Industrial Safety**: Monitoring operator fatigue and environmental hazards

## Customization and Extension

### Adding New Observable States
```cpp
// Extend observable state dimension
.SetObservableStateDim(15)  // Add 3 new sensors

// Update input creation
Tensor observable_state({15});
observable_state.SetValue({
    // ... existing 12 values ...
    new_sensor_1_value,
    new_sensor_2_value, 
    new_sensor_3_value
});
```

### Custom Meta-Evaluation Functions
```cpp
class CustomMetaEvaluation : public MetaEvaluationFunction {
public:
    float Evaluate(const Tensor& predicted_state,
                  const Tensor& uncertainty,
                  const TimeInfo& context) const override {
        // Custom evaluation logic
        float concentration = predicted_state.GetValue(0);
        float stress = predicted_state.GetValue(1);
        
        // Example: Prioritize concentration during work hours
        float time_weight = IsWorkingHours(context) ? 2.0f : 1.0f;
        return time_weight * concentration - 0.5f * stress;
    }
    
    // ... implement other required methods
};
```

## Citation

If you use this STA implementation in your research, please cite:

```bibtex
@article{sta_architecture_2025,
  title={空気を読めるアーキテクチャ：任意入力に対するリアルタイム状態予測と感受性制御に基づく適応的ヒューマン・インタフェース},
  author={Your Name},
  journal={Journal of Human-Computer Interaction},
  year={2025},
  note={Implementation available at: https://github.com/yourusername/LwTT}
}
```

## License

This example is provided under the same license as the LwTT library (MIT License).

---

For more information about the LwTT library and additional examples, visit the [main documentation](../../README.md).
