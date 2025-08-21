#ifndef ANALOG_UNIVERSAL_NODE_ENGINE_AVX2_H
#define ANALOG_UNIVERSAL_NODE_ENGINE_AVX2_H

#include <cstdint>
#include <cmath>
#include <immintrin.h>
#include <vector>

// Helper function to clamp values
template <typename T>
T clamp_custom(T value, T min, T max) {
    if (value < min) return min;
    if (value > max) return max;
    return value;
}

// Global metrics structure
struct EngineMetrics {
    uint64_t total_execution_time_ns;
    uint64_t avx2_operation_time_ns;
    uint64_t total_operations;
    uint64_t avx2_operations;
    uint64_t node_processes;
    uint64_t harmonic_generations;
    double current_ns_per_op;
    double current_ops_per_second;
    double speedup_factor;
    const double target_ns_per_op = 8000.0;

    void reset();
    void update_performance();
    void print_metrics();
};

// AVX2 Vectorized Math Functions
namespace AVX2Math {
    __m256 fast_sin_avx2(__m256 x);
    __m256 fast_cos_avx2(__m256 x);
    void generate_harmonics_avx2(float input_signal, float pass_offset, float* harmonics_out);
    float process_spectral_avx2(float output_base);
}

// Analog Universal Node optimized with AVX2
class AnalogUniversalNodeAVX2 {
private:
    double integrator_state = 0.0;
    double feedback_gain = 0.0;
    double current_output = 0.0;
    double previous_input = 0.0;
    uint64_t operation_count = 0;

public:
    // For 3D cellular automata
    int16_t x, y, z;
    uint16_t node_id;

    AnalogUniversalNodeAVX2() = default;

    double processSignalAVX2(double input_signal, double control_signal, double aux_signal);
    double processSignal(double input_signal, double control_signal, double aux_signal);
    void setFeedback(double feedback_coefficient);
    double getOutput() const;
    double getIntegratorState() const;
    void resetIntegrator();
};

// D-ASE cellular engine with AVX2 optimizations
class AnalogCellularEngineAVX2 {
private:
    std::vector<AnalogUniversalNodeAVX2> nodes;
    double system_frequency;
    double noise_level;

public:
    AnalogCellularEngineAVX2(size_t num_nodes);
    double processSignalWaveAVX2(double input_signal, double control_pattern);
    double performSignalSweepAVX2(double frequency);
    void runBuiltinBenchmark(int iterations);
    EngineMetrics getMetrics() const;
    void printLiveMetrics();
    void resetMetrics();
    
    // This function is now public
    double generateNoiseSignal();
    double calculateInterNodeCoupling(size_t node_index);
};

// CPU feature detection utility
class CPUFeatures {
public:
    static bool hasAVX2();
    static bool hasFMA();
    static bool checkCPUID(int function, int subfunction, int reg, int bit);
    static void printCapabilities();
};

#endif // ANALOG_UNIVERSAL_NODE_ENGINE_AVX2_H