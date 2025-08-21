#!/usr/bin/env python3
"""
DASE Python Interface Layer
Bridges the C++ engine with the DVSL web interface
"""

import numpy as np
import json
import time
import threading
from typing import Dict, List, Tuple, Optional, Any
import logging

try:
    import dase_engine  # Our compiled C++ module
except ImportError:
    print("Warning: dase_engine module not found. Falling back to simulation mode.")
    dase_engine = None

class DVSLSymbolProcessor:
    """Processes DVSL symbols and converts them to engine operations"""
    
    SYMBOL_MAP = {
        'amp': 'amplifier',
        'integrator': 'integrator', 
        'summer': 'summer',
        'derivative': 'differentiator',
        'multiplier': 'multiplier',
        'comparator': 'comparator',
        'sine_osc': 'sine_oscillator',
        'square_osc': 'square_oscillator',
        'noise_gen': 'noise_generator',
        'mw_osc': 'microwave_oscillator',
        'waveguide': 'waveguide',
        'neural_coupler': 'neural_coupler',
        'resonant_cavity': 'resonant_cavity',
        'neuron': 'neuron',
        'synaptic': 'synapse'
    }
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def parse_formula(self, formula: str) -> Dict[str, Any]:
        """Parse DVSL formula into executable components"""
        if not formula.startswith('='):
            return {'type': 'value', 'value': float(formula) if formula.replace('.','').isdigit() else 0.0}
        
        # Remove = and parse
        expr = formula[1:].strip()
        
        # Simple parser for basic DVSL syntax
        if '(' in expr:
            func_name = expr.split('(')[0]
            params_str = expr[expr.find('(')+1:expr.rfind(')')]
            params = self._parse_parameters(params_str)
            
            return {
                'type': 'function',
                'function': self.SYMBOL_MAP.get(func_name, func_name),
                'parameters': params
            }
        else:
            return {'type': 'reference', 'cell': expr}
    
    def _parse_parameters(self, params_str: str) -> Dict[str, Any]:
        """Parse parameter string into dictionary"""
        params = {}
        if not params_str.strip():
            return params
            
        parts = params_str.split(',')
        for part in parts:
            part = part.strip()
            if '=' in part:
                key, value = part.split('=', 1)
                key = key.strip()
                value = value.strip()
                try:
                    params[key] = float(value)
                except ValueError:
                    params[key] = value
            else:
                # Positional parameter (cell reference)
                if not 'inputs' in params:
                    params['inputs'] = []
                params['inputs'].append(part)
        
        return params

class DASEEngine:
    """High-level Python interface to the DASE engine"""
    
    def __init__(self, num_nodes: int = 1000):
        self.num_nodes = num_nodes
        self.engine = None
        self.nodes = {}
        self.cell_values = {}
        self.symbol_processor = DVSLSymbolProcessor()
        self.is_initialized = False
        self.simulation_running = False
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.performance_metrics = {
            'operations_per_second': 0,
            'total_operations': 0,
            'avg_latency_ns': 0,
            'last_update': time.time()
        }
    
    def initialize(self) -> bool:
        """Initialize the C++ engine"""
        if dase_engine is None:
            self.logger.warning("DASE engine not available, using simulation mode")
            return False
            
        try:
            # Check CPU capabilities
            if not dase_engine.CPUFeatures.has_avx2():
                self.logger.warning("AVX2 not supported, performance may be reduced")
            
            # Initialize engine
            self.engine = dase_engine.AnalogCellularEngine(self.num_nodes)
            self.is_initialized = True
            self.logger.info(f"DASE engine initialized with {self.num_nodes} nodes")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize DASE engine: {e}")
            return False
    
    def process_cell_data(self, cell_data: Dict[str, Any]) -> Dict[str, float]:
        """Process all cell data and return computed values"""
        if not self.is_initialized:
            return self._simulate_processing(cell_data)
        
        results = {}
        start_time = time.perf_counter()
        
        try:
            # Build dependency graph
            dependency_graph = self._build_dependency_graph(cell_data)
            
            # Process cells in topological order
            for cell_id in self._topological_sort(dependency_graph):
                if cell_id in cell_data:
                    result = self._process_single_cell(cell_id, cell_data)
                    results[cell_id] = result
                    self.cell_values[cell_id] = result
            
            # Update performance metrics
            elapsed_time = time.perf_counter() - start_time
            self._update_performance_metrics(len(results), elapsed_time)
            
        except Exception as e:
            self.logger.error(f"Error processing cell data: {e}")
            results = self._simulate_processing(cell_data)
        
        return results
    
    def _process_single_cell(self, cell_id: str, cell_data: Dict[str, Any]) -> float:
        """Process a single cell and return its value"""
        cell = cell_data[cell_id]
        
        if 'formula' in cell and cell['formula']:
            parsed = self.symbol_processor.parse_formula(cell['formula'])
            return self._execute_operation(parsed, cell_data)
        elif 'value' in cell and cell['value']:
            try:
                return float(cell['value'])
            except (ValueError, TypeError):
                return 0.0
        else:
            return 0.0
    
    def _execute_operation(self, operation: Dict[str, Any], cell_data: Dict[str, Any]) -> float:
        """Execute a parsed operation"""
        if operation['type'] == 'value':
            return operation['value']
        elif operation['type'] == 'reference':
            cell_ref = operation['cell']
            return self.cell_values.get(cell_ref, 0.0)
        elif operation['type'] == 'function':
            return self._execute_function(operation, cell_data)
        else:
            return 0.0
    
    def _execute_function(self, operation: Dict[str, Any], cell_data: Dict[str, Any]) -> float:
        """Execute a function operation using the C++ engine"""
        func_name = operation['function']
        params = operation['parameters']
        
        # Get input values
        inputs = []
        if 'inputs' in params:
            for input_ref in params['inputs']:
                inputs.append(self.cell_values.get(input_ref, 0.0))
        
        # Process based on function type
        if func_name == 'amplifier':
            gain = params.get('gain', 1.0)
            input_val = inputs[0] if inputs else 0.0
            return input_val * gain
            
        elif func_name == 'integrator':
            dt = params.get('dt', 0.01)
            input_val = inputs[0] if inputs else 0.0
            # Use engine for integration if available
            if self.engine:
                node = dase_engine.AnalogUniversalNode()
                return node.process_signal_avx2(input_val, 0.0, dt)
            else:
                return input_val * dt
                
        elif func_name == 'summer':
            return sum(inputs)
            
        elif func_name == 'multiplier':
            if len(inputs) >= 2:
                return inputs[0] * inputs[1]
            return 0.0
            
        elif func_name == 'sine_oscillator':
            freq = params.get('freq', 440.0)
            amp = params.get('amp', 1.0)
            if dase_engine:
                t = time.time() % (1.0 / freq) if freq > 0 else 0
                return amp * dase_engine.fast_sin_scalar(2 * np.pi * freq * t)
            else:
                t = time.time()
                return amp * np.sin(2 * np.pi * freq * t)
                
        elif func_name == 'noise_generator':
            amp = params.get('amp', 0.1)
            if self.engine:
                return self.engine.generate_noise_signal() * amp
            else:
                return np.random.normal(0, amp)
        
        # Add more function implementations as needed
        return 0.0
    
    def _build_dependency_graph(self, cell_data: Dict[str, Any]) -> Dict[str, List[str]]:
        """Build dependency graph for cells"""
        graph = {}
        
        for cell_id, cell in cell_data.items():
            dependencies = []
            if 'formula' in cell and cell['formula']:
                parsed = self.symbol_processor.parse_formula(cell['formula'])
                if parsed['type'] == 'function' and 'inputs' in parsed['parameters']:
                    dependencies.extend(parsed['parameters']['inputs'])
                elif parsed['type'] == 'reference':
                    dependencies.append(parsed['cell'])
            
            graph[cell_id] = dependencies
        
        return graph
    
    def _topological_sort(self, graph: Dict[str, List[str]]) -> List[str]:
        """Perform topological sort on dependency graph"""
        # Simple topological sort implementation
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(node):
            if node in temp_visited:
                return  # Cycle detected, skip
            if node in visited:
                return
                
            temp_visited.add(node)
            for dependency in graph.get(node, []):
                if dependency in graph:
                    visit(dependency)
            temp_visited.remove(node)
            visited.add(node)
            result.append(node)
        
        for node in graph:
            if node not in visited:
                visit(node)
        
        return result
    
    def _simulate_processing(self, cell_data: Dict[str, Any]) -> Dict[str, float]:
        """Fallback simulation when C++ engine is not available"""
        results = {}
        for cell_id, cell in cell_data.items():
            if 'value' in cell and cell['value']:
                try:
                    results[cell_id] = float(cell['value'])
                except (ValueError, TypeError):
                    results[cell_id] = 0.0
            else:
                results[cell_id] = np.random.random() * 10 - 5  # Random simulation
        return results
    
    def _update_performance_metrics(self, num_operations: int, elapsed_time: float):
        """Update performance tracking metrics"""
        self.performance_metrics['total_operations'] += num_operations
        self.performance_metrics['avg_latency_ns'] = elapsed_time * 1e9 / num_operations
        
        current_time = time.time()
        time_delta = current_time - self.performance_metrics['last_update']
        if time_delta > 0:
            self.performance_metrics['operations_per_second'] = num_operations / time_delta
        self.performance_metrics['last_update'] = current_time
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        if self.engine and hasattr(self.engine, 'get_metrics'):
            cpp_metrics = self.engine.get_metrics()
            return {
                'cpp_metrics': {
                    'total_operations': cpp_metrics.total_operations,
                    'avx2_operations': cpp_metrics.avx2_operations,
                    'current_ns_per_op': cpp_metrics.current_ns_per_op,
                    'current_ops_per_second': cpp_metrics.current_ops_per_second,
                    'speedup_factor': cpp_metrics.speedup_factor
                },
                'python_metrics': self.performance_metrics
            }
        else:
            return {'python_metrics': self.performance_metrics}
    
    def run_benchmark(self, iterations: int = 1000) -> Dict[str, Any]:
        """Run performance benchmark"""
        if not self.is_initialized:
            return {'error': 'Engine not initialized'}
        
        try:
            self.engine.run_builtin_benchmark(iterations)
            return self.get_performance_metrics()
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            return {'error': str(e)}
    
    def start_real_time_processing(self, cell_data: Dict[str, Any], update_callback=None):
        """Start real-time processing loop"""
        if self.simulation_running:
            return
        
        self.simulation_running = True
        
        def processing_loop():
            while self.simulation_running:
                try:
                    results = self.process_cell_data(cell_data)
                    if update_callback:
                        update_callback(results)
                    time.sleep(0.01)  # 100Hz update rate
                except Exception as e:
                    self.logger.error(f"Error in processing loop: {e}")
                    break
        
        self.processing_thread = threading.Thread(target=processing_loop, daemon=True)
        self.processing_thread.start()
    
    def stop_real_time_processing(self):
        """Stop real-time processing loop"""
        self.simulation_running = False
        if hasattr(self, 'processing_thread'):
            self.processing_thread.join(timeout=1.0)

# Global engine instance
_engine_instance = None

def get_engine(num_nodes: int = 1000) -> DASEEngine:
    """Get or create global engine instance"""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = DASEEngine(num_nodes)
        _engine_instance.initialize()
    return _engine_instance
