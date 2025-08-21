#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "analog_universal_node_engine_avx2.h"

namespace py = pybind11;

PYBIND11_MODULE(dase_engine, m) {
    m.doc() = "DASE Analog Engine - High-performance analog signal processing with AVX2 optimization";

    // EngineMetrics struct
    py::class_<EngineMetrics>(m, "EngineMetrics")
        .def(py::init<>())
        .def_readwrite("total_execution_time_ns", &EngineMetrics::total_execution_time_ns)
        .def_readwrite("avx2_operation_time_ns", &EngineMetrics::avx2_operation_time_ns)
        .def_readwrite("total_operations", &EngineMetrics::total_operations)
        .def_readwrite("avx2_operations", &EngineMetrics::avx2_operations)
        .def_readwrite("node_processes", &EngineMetrics::node_processes)
        .def_readwrite("harmonic_generations", &EngineMetrics::harmonic_generations)
        .def_readonly("current_ns_per_op", &EngineMetrics::current_ns_per_op)
        .def_readonly("current_ops_per_second", &EngineMetrics::current_ops_per_second)
        .def_readonly("speedup_factor", &EngineMetrics::speedup_factor)
        .def_readonly("target_ns_per_op", &EngineMetrics::target_ns_per_op)
        .def("reset", &EngineMetrics::reset)
        .def("update_performance", &EngineMetrics::update_performance)
        .def("print_metrics", &EngineMetrics::print_metrics);

    // AnalogUniversalNodeAVX2 class
    py::class_<AnalogUniversalNodeAVX2>(m, "AnalogUniversalNode")
        .def(py::init<>())
        .def_readwrite("x", &AnalogUniversalNodeAVX2::x)
        .def_readwrite("y", &AnalogUniversalNodeAVX2::y) 
        .def_readwrite("z", &AnalogUniversalNodeAVX2::z)
        .def_readwrite("node_id", &AnalogUniversalNodeAVX2::node_id)
        .def("process_signal", &AnalogUniversalNodeAVX2::processSignal,
             "Process analog signal through the node",
             py::arg("input_signal"), py::arg("control_signal"), py::arg("aux_signal"))
        .def("process_signal_avx2", &AnalogUniversalNodeAVX2::processSignalAVX2,
             "Process analog signal with AVX2 optimization",
             py::arg("input_signal"), py::arg("control_signal"), py::arg("aux_signal"))
        .def("set_feedback", &AnalogUniversalNodeAVX2::setFeedback,
             "Set feedback coefficient",
             py::arg("feedback_coefficient"))
        .def("get_output", &AnalogUniversalNodeAVX2::getOutput,
             "Get current output value")
        .def("get_integrator_state", &AnalogUniversalNodeAVX2::getIntegratorState,
             "Get current integrator state")
        .def("reset_integrator", &AnalogUniversalNodeAVX2::resetIntegrator,
             "Reset integrator state to zero");

    // AnalogCellularEngineAVX2 class
    py::class_<AnalogCellularEngineAVX2>(m, "AnalogCellularEngine")
        .def(py::init<size_t>(), "Initialize engine with specified number of nodes",
             py::arg("num_nodes"))
        .def("process_signal_wave", &AnalogCellularEngineAVX2::processSignalWaveAVX2,
             "Process signal wave through cellular array",
             py::arg("input_signal"), py::arg("control_pattern"))
        .def("perform_signal_sweep", &AnalogCellularEngineAVX2::performSignalSweepAVX2,
             "Perform frequency sweep operation",
             py::arg("frequency"))
        .def("run_builtin_benchmark", &AnalogCellularEngineAVX2::runBuiltinBenchmark,
             "Run performance benchmark",
             py::arg("iterations") = 1000)
        .def("get_metrics", &AnalogCellularEngineAVX2::getMetrics,
             "Get current performance metrics")
        .def("print_live_metrics", &AnalogCellularEngineAVX2::printLiveMetrics,
             "Print current performance metrics")
        .def("reset_metrics", &AnalogCellularEngineAVX2::resetMetrics,
             "Reset performance counters")
        .def("generate_noise_signal", &AnalogCellularEngineAVX2::generateNoiseSignal,
             "Generate random noise signal")
        .def("calculate_inter_node_coupling", &AnalogCellularEngineAVX2::calculateInterNodeCoupling,
             "Calculate coupling between nodes",
             py::arg("node_index"));

    // CPUFeatures utility class
    py::class_<CPUFeatures>(m, "CPUFeatures")
        .def_static("has_avx2", &CPUFeatures::hasAVX2,
                   "Check if CPU supports AVX2 instructions")
        .def_static("has_fma", &CPUFeatures::hasFMA,
                   "Check if CPU supports FMA instructions")
        .def_static("print_capabilities", &CPUFeatures::printCapabilities,
                   "Print detected CPU capabilities");

    // AVX2Math namespace functions
    m.def("fast_sin_scalar", [](float x) {
        // Wrapper for scalar version of fast sin
        __m256 input = _mm256_set1_ps(x);
        __m256 result = AVX2Math::fast_sin_avx2(input);
        float output[8];
        _mm256_store_ps(output, result);
        return output[0];
    }, "Fast sine approximation (scalar version)", py::arg("x"));

    m.def("fast_cos_scalar", [](float x) {
        // Wrapper for scalar version of fast cos
        __m256 input = _mm256_set1_ps(x);
        __m256 result = AVX2Math::fast_cos_avx2(input);
        float output[8];
        _mm256_store_ps(output, result);
        return output[0];
    }, "Fast cosine approximation (scalar version)", py::arg("x"));

    m.def("generate_harmonics", [](float input_signal, float pass_offset) {
        // Generate harmonics and return as Python list
        float harmonics[8];
        AVX2Math::generate_harmonics_avx2(input_signal, pass_offset, harmonics);
        std::vector<float> result(harmonics, harmonics + 8);
        return result;
    }, "Generate 8 harmonic components", py::arg("input_signal"), py::arg("pass_offset"));

    m.def("process_spectral", &AVX2Math::process_spectral_avx2,
          "Process spectral components with AVX2",
          py::arg("output_base"));

    // Utility functions for batch processing
    m.def("process_signal_batch", [](AnalogCellularEngineAVX2& engine, 
                                     const std::vector<double>& input_signals,
                                     const std::vector<double>& control_patterns) {
        std::vector<double> results;
        size_t min_size = std::min(input_signals.size(), control_patterns.size());
        results.reserve(min_size);
        
        for (size_t i = 0; i < min_size; ++i) {
            double result = engine.processSignalWaveAVX2(input_signals[i], control_patterns[i]);
            results.push_back(result);
        }
        return results;
    }, "Process batch of signals through engine",
       py::arg("engine"), py::arg("input_signals"), py::arg("control_patterns"));

    m.def("frequency_sweep_batch", [](AnalogCellularEngineAVX2& engine,
                                      const std::vector<double>& frequencies) {
        std::vector<double> results;
        results.reserve(frequencies.size());
        
        for (double freq : frequencies) {
            double result = engine.performSignalSweepAVX2(freq);
            results.push_back(result);
        }
        return results;
    }, "Perform frequency sweep over multiple frequencies",
       py::arg("engine"), py::arg("frequencies"));

    // Version and build information
    m.attr("__version__") = "1.0.0";
    m.attr("avx2_enabled") = true;
    
    #ifdef _OPENMP
    m.attr("openmp_enabled") = true;
    #else
    m.attr("openmp_enabled") = false;
    #endif
}