#!/usr/bin/env python3
"""
DASE Web Server Interface
Provides REST API endpoints for the DVSL web interface
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import json
import os
import time
import logging
from typing import Dict, Any
import threading

from dase_interface import get_engine, DASEEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for web interface

# Global state
current_cell_data = {}
engine = None
real_time_active = False

@app.route('/')
def serve_interface():
    """Serve the main DVSL interface"""
    return send_file('dvsl_interface_v2.html')

@app.route('/<filename>')
def serve_root_files(filename):
    """Serve files from root directory"""
    try:
        return send_from_directory('.', filename)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('.', filename)

@app.route('/api/initialize', methods=['POST'])
def initialize_engine():
    """Initialize the DASE engine"""
    global engine
    try:
        data = request.get_json() or {}
        num_nodes = data.get('num_nodes', 1000)
        
        engine = get_engine(num_nodes)
        
        # Get CPU capabilities
        cpu_info = {
            'avx2_supported': True,  # Will be checked by C++ module
            'num_nodes': num_nodes,
            'initialized': engine.is_initialized
        }
        
        logger.info(f"Engine initialized with {num_nodes} nodes")
        return jsonify({
            'success': True,
            'message': 'Engine initialized successfully',
            'cpu_info': cpu_info
        })
        
    except Exception as e:
        logger.error(f"Failed to initialize engine: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/process_cells', methods=['POST'])
def process_cells():
    """Process cell data and return computed values"""
    global current_cell_data, engine
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        cell_data = data.get('cells', {})
        current_cell_data = cell_data
        
        if engine is None:
            engine = get_engine()
        
        # Process cells
        results = engine.process_cell_data(cell_data)
        
        return jsonify({
            'success': True,
            'results': results,
            'metrics': engine.get_performance_metrics()
        })
        
    except Exception as e:
        logger.error(f"Error processing cells: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/validate_formula', methods=['POST'])
def validate_formula():
    """Validate a DVSL formula"""
    global engine
    try:
        data = request.get_json()
        formula = data.get('formula', '')
        
        if not formula:
            return jsonify({'valid': False, 'error': 'Empty formula'})
        
        # Basic validation
        if not formula.startswith('='):
            return jsonify({'valid': False, 'error': 'Formula must start with ='})
        
        # Try to parse the formula
        if engine is None:
            engine = get_engine()
        
        try:
            parsed = engine.symbol_processor.parse_formula(formula)
            return jsonify({
                'valid': True,
                'parsed': parsed,
                'message': 'Formula is valid'
            })
        except Exception as e:
            return jsonify({
                'valid': False,
                'error': f'Parse error: {str(e)}'
            })
            
    except Exception as e:
        logger.error(f"Error validating formula: {e}")
        return jsonify({
            'valid': False,
            'error': str(e)
        }), 500

@app.route('/api/simulation/start', methods=['POST'])
def start_simulation():
    """Start real-time simulation"""
    global real_time_active, engine, current_cell_data
    
    try:
        if real_time_active:
            return jsonify({'success': False, 'error': 'Simulation already running'})
        
        if engine is None:
            engine = get_engine()
        
        def update_callback(results):
            # In a real implementation, this would push updates to connected clients
            # via WebSocket or Server-Sent Events
            pass
        
        engine.start_real_time_processing(current_cell_data, update_callback)
        real_time_active = True
        
        logger.info("Real-time simulation started")
        return jsonify({
            'success': True,
            'message': 'Simulation started'
        })
        
    except Exception as e:
        logger.error(f"Error starting simulation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/simulation/stop', methods=['POST'])
def stop_simulation():
    """Stop real-time simulation"""
    global real_time_active, engine
    
    try:
        if not real_time_active:
            return jsonify({'success': False, 'error': 'Simulation not running'})
        
        if engine:
            engine.stop_real_time_processing()
        
        real_time_active = False
        
        logger.info("Real-time simulation stopped")
        return jsonify({
            'success': True,
            'message': 'Simulation stopped'
        })
        
    except Exception as e:
        logger.error(f"Error stopping simulation: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/simulation/status', methods=['GET'])
def get_simulation_status():
    """Get current simulation status"""
    global real_time_active, engine
    
    try:
        status = {
            'running': real_time_active,
            'initialized': engine is not None and engine.is_initialized,
            'active_cells': len(current_cell_data),
        }
        
        if engine:
            status['metrics'] = engine.get_performance_metrics()
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error getting simulation status: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/benchmark', methods=['POST'])
def run_benchmark():
    """Run performance benchmark"""
    global engine
    
    try:
        data = request.get_json() or {}
        iterations = data.get('iterations', 1000)
        
        if engine is None:
            engine = get_engine()
        
        if not engine.is_initialized:
            return jsonify({
                'success': False,
                'error': 'Engine not initialized'
            }), 400
        
        logger.info(f"Starting benchmark with {iterations} iterations")
        results = engine.run_benchmark(iterations)
        
        return jsonify({
            'success': True,
            'benchmark_results': results
        })
        
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get current performance metrics"""
    global engine
    
    try:
        if engine is None:
            return jsonify({'error': 'Engine not initialized'}), 400
        
        metrics = engine.get_performance_metrics()
        return jsonify(metrics)
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/cell/<cell_id>', methods=['GET'])
def get_cell_value(cell_id):
    """Get value for a specific cell"""
    global engine
    try:
        if engine and cell_id in engine.cell_values:
            return jsonify({
                'cell_id': cell_id,
                'value': engine.cell_values[cell_id],
                'timestamp': time.time()
            })
        else:
            return jsonify({
                'cell_id': cell_id,
                'value': 0.0,
                'timestamp': time.time()
            })
            
    except Exception as e:
        logger.error(f"Error getting cell value: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/cell/<cell_id>', methods=['PUT'])
def update_cell_value(cell_id):
    """Update value for a specific cell"""
    global current_cell_data, engine
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Update cell data
        if cell_id not in current_cell_data:
            current_cell_data[cell_id] = {}
        
        current_cell_data[cell_id].update(data)
        
        # Reprocess if engine is available
        if engine:
            results = engine.process_cell_data(current_cell_data)
            return jsonify({
                'success': True,
                'cell_id': cell_id,
                'value': results.get(cell_id, 0.0),
                'all_results': results
            })
        else:
            return jsonify({
                'success': True,
                'cell_id': cell_id,
                'message': 'Cell updated (engine not available)'
            })
            
    except Exception as e:
        logger.error(f"Error updating cell value: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/export', methods=['GET'])
def export_project():
    """Export current project as JSON"""
    global engine
    try:
        project_data = {
            'version': '1.0',
            'timestamp': time.time(),
            'cells': current_cell_data,
            'metrics': engine.get_performance_metrics() if engine else None
        }
        
        return jsonify(project_data)
        
    except Exception as e:
        logger.error(f"Error exporting project: {e}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/import', methods=['POST'])
def import_project():
    """Import project from JSON"""
    global current_cell_data, engine
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate project data
        if 'cells' not in data:
            return jsonify({'error': 'Invalid project format'}), 400
        
        current_cell_data = data['cells']
        
        # Reprocess with engine if available
        if engine:
            results = engine.process_cell_data(current_cell_data)
            return jsonify({
                'success': True,
                'message': 'Project imported successfully',
                'results': results
            })
        else:
            return jsonify({
                'success': True,
                'message': 'Project imported (engine not available)'
            })
            
    except Exception as e:
        logger.error(f"Error importing project: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

def run_server(host='localhost', port=5000, debug=False):
    """Run the web server"""
    logger.info(f"Starting DASE web server on http://{host}:{port}")
    app.run(host=host, port=port, debug=debug, threaded=True)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='DASE Web Server')
    parser.add_argument('--host', default='localhost', help='Server host')
    parser.add_argument('--port', type=int, default=5000, help='Server port')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    run_server(args.host, args.port, args.debug)