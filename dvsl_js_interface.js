/**
 * DVSL JavaScript Interface
 * Connects the web interface to the Python DASE engine via REST API
 */

class EnhancedDVSLEngine {
    constructor() {
        this.isInitialized = false;
        this.realTimeMode = false;
        this.baseUrl = 'http://localhost:5000/api';
        this.cellData = {};
        this.intervalId = null;
        this.wsConnection = null;
        
        // Performance tracking
        this.metrics = {
            operations: 0,
            errors: 0,
            lastUpdate: Date.now()
        };
    }

    async initialize() {
        try {
            console.log('Initializing DASE engine...');
            
            const response = await fetch(`${this.baseUrl}/initialize`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    num_nodes: 1000
                })
            });

            const result = await response.json();
            
            if (result.success) {
                this.isInitialized = true;
                console.log('DASE engine initialized successfully');
                console.log('CPU Info:', result.cpu_info);
                return true;
            } else {
                console.error('Failed to initialize engine:', result.error);
                return false;
            }
        } catch (error) {
            console.error('Error initializing engine:', error);
            return false;
        }
    }

    async processCells(cellData) {
        try {
            this.cellData = cellData;
            
            const response = await fetch(`${this.baseUrl}/process_cells`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    cells: cellData
                })
            });

            const result = await response.json();
            
            if (result.success) {
                this.metrics.operations++;
                return result.results;
            } else {
                this.metrics.errors++;
                console.error('Cell processing failed:', result.error);
                return {};
            }
        } catch (error) {
            this.metrics.errors++;
            console.error('Error processing cells:', error);
            return {};
        }
    }

    async validateFormula(formula) {
        try {
            const response = await fetch(`${this.baseUrl}/validate_formula`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    formula: formula
                })
            });

            const result = await response.json();
            return result;
        } catch (error) {
            console.error('Error validating formula:', error);
            return { valid: false, error: 'Validation failed' };
        }
    }

    async startRealTimeMode(cellData, updateCallback) {
        if (this.realTimeMode) {
            console.warn('Real-time mode already active');
            return;
        }

        try {
            // Start server-side simulation
            const response = await fetch(`${this.baseUrl}/simulation/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    cells: cellData
                })
            });

            const result = await response.json();
            
            if (result.success) {
                this.realTimeMode = true;
                this.cellData = cellData;
                
                // Start polling for updates
                this.intervalId = setInterval(async () => {
                    try {
                        const results = await this.processCells(this.cellData);
                        if (updateCallback) {
                            updateCallback(results);
                        }
                    } catch (error) {
                        console.error('Error in real-time update:', error);
                    }
                }, 100); // 10Hz update rate
                
                console.log('Real-time mode started');
            } else {
                console.error('Failed to start simulation:', result.error);
            }
        } catch (error) {
            console.error('Error starting real-time mode:', error);
        }
    }

    async stopRealTimeMode() {
        if (!this.realTimeMode) {
            return;
        }

        try {
            // Stop server-side simulation
            const response = await fetch(`${this.baseUrl}/simulation/stop`, {
                method: 'POST'
            });

            const result = await response.json();
            
            if (result.success) {
                this.realTimeMode = false;
                
                // Stop polling
                if (this.intervalId) {
                    clearInterval(this.intervalId);
                    this.intervalId = null;
                }
                
                console.log('Real-time mode stopped');
            } else {
                console.error('Failed to stop simulation:', result.error);
            }
        } catch (error) {
            console.error('Error stopping real-time mode:', error);
        }
    }

    async getMetrics() {
        try {
            const response = await fetch(`${this.baseUrl}/metrics`);
            const result = await response.json();
            return result;
        } catch (error) {
            console.error('Error getting metrics:', error);
            return this.metrics;
        }
    }

    async runBenchmark(iterations = 1000) {
        try {
            console.log(`Running benchmark with ${iterations} iterations...`);
            
            const response = await fetch(`${this.baseUrl}/benchmark`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    iterations: iterations
                })
            });

            const result = await response.json();
            
            if (result.success) {
                console.log('Benchmark completed:', result.benchmark_results);
                return result.benchmark_results;
            } else {
                console.error('Benchmark failed:', result.error);
                return null;
            }
        } catch (error) {
            console.error('Error running benchmark:', error);
            return null;
        }
    }

    async updateCell(cellId, cellData) {
        try {
            const response = await fetch(`${this.baseUrl}/cell/${cellId}`, {
                method: 'PUT',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(cellData)
            });

            const result = await response.json();
            return result;
        } catch (error) {
            console.error('Error updating cell:', error);
            return { success: false, error: error.message };
        }
    }

    async getCellValue(cellId) {
        try {
            const response = await fetch(`${this.baseUrl}/cell/${cellId}`);
            const result = await response.json();
            return result.value || 0.0;
        } catch (error) {
            console.error('Error getting cell value:', error);
            return 0.0;
        }
    }

    async exportProject() {
        try {
            const response = await fetch(`${this.baseUrl}/export`);
            const result = await response.json();
            return result;
        } catch (error) {
            console.error('Error exporting project:', error);
            return null;
        }
    }

    async importProject(projectData) {
        try {
            const response = await fetch(`${this.baseUrl}/import`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(projectData)
            });

            const result = await response.json();
            return result;
        } catch (error) {
            console.error('Error importing project:', error);
            return { success: false, error: error.message };
        }
    }

    async getSimulationStatus() {
        try {
            const response = await fetch(`${this.baseUrl}/simulation/status`);
            const result = await response.json();
            return result;
        } catch (error) {
            console.error('Error getting simulation status:', error);
            return { running: false, error: error.message };
        }
    }
}

// Global simulation state
let simulationState = {
    running: false,
    paused: false,
    timeStep: 10,
    speed: 1.0,
    startTime: 0,
    currentTime: 0,
    iterations: 0
};

// Enhanced simulation controls
function initializeSimulationControls() {
    // Time step controls
    const timeStepInput = document.getElementById('timeStepInput');
    const timeStepValue = document.getElementById('timeStepValue');
    
    if (timeStepInput && timeStepValue) {
        timeStepInput.addEventListener('input', (e) => {
            timeStepValue.value = e.target.value;
            simulationState.timeStep = parseInt(e.target.value);
        });
        
        timeStepValue.addEventListener('input', (e) => {
            timeStepInput.value = e.target.value;
            simulationState.timeStep = parseInt(e.target.value);
        });
    }
    
    // Speed controls
    const speedInput = document.getElementById('speedInput');
    const speedValue = document.getElementById('speedValue');
    
    if (speedInput && speedValue) {
        speedInput.addEventListener('input', (e) => {
            speedValue.value = e.target.value;
            simulationState.speed = parseFloat(e.target.value);
        });
        
        speedValue.addEventListener('input', (e) => {
            speedInput.value = e.target.value;
            simulationState.speed = parseFloat(e.target.value);
        });
    }
    
    // Start performance monitoring
    updateSimulationMetrics();
}

function updateSimulationMetrics() {
    if (!simulationState.running) {
        setTimeout(updateSimulationMetrics, 1000);
        return;
    }
    
    const now = Date.now();
    const elapsed = (now - simulationState.startTime) / 1000;
    
    // Update display elements
    const simTime = document.getElementById('simTime');
    const realTime = document.getElementById('realTime');
    const iterations = document.getElementById('iterations');
    const opsPerSec = document.getElementById('opsPerSec');
    const activeCells = document.getElementById('activeCells');
    
    if (simTime) simTime.textContent = `${elapsed.toFixed(2)}s`;
    if (realTime) realTime.textContent = `${elapsed.toFixed(2)}s`;
    if (iterations) iterations.textContent = simulationState.iterations;
    if (opsPerSec) opsPerSec.textContent = Math.round(simulationState.iterations / elapsed);
    if (activeCells) activeCells.textContent = getActiveCells().length;
    
    setTimeout(updateSimulationMetrics, 100);
}

// Enhanced terminal system
function initializeTerminal() {
    const terminalInput = document.getElementById('terminalInput');
    const terminalContent = document.getElementById('terminalContent');
    
    if (!terminalInput || !terminalContent) return;
    
    terminalInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            const command = terminalInput.value.trim();
            if (command) {
                processTerminalCommand(command);
                terminalInput.value = '';
            }
        }
    });
    
    addTerminalLine('DASE Terminal ready. Type "help" for commands.', 'terminal-success');
}

function addTerminalLine(text, className = '') {
    const terminalContent = document.getElementById('terminalContent');
    if (!terminalContent) return;
    
    const line = document.createElement('div');
    line.className = `terminal-line ${className}`;
    line.textContent = text;
    
    // Insert before the input line
    const inputLine = terminalContent.querySelector('.terminal-input-line');
    if (inputLine) {
        terminalContent.insertBefore(line, inputLine);
    } else {
        terminalContent.appendChild(line);
    }
    
    // Scroll to bottom
    terminalContent.scrollTop = terminalContent.scrollHeight;
}

async function processTerminalCommand(command) {
    addTerminalLine(`dase$ ${command}`, 'terminal-prompt');
    
    const parts = command.split(' ');
    const cmd = parts[0].toLowerCase();
    const args = parts.slice(1);
    
    try {
        switch (cmd) {
            case 'help':
                showTerminalHelp();
                break;
                
            case 'status':
                await showSystemStatus();
                break;
                
            case 'benchmark':
                const iterations = parseInt(args[0]) || 1000;
                await runTerminalBenchmark(iterations);
                break;
                
            case 'metrics':
                await showPerformanceMetrics();
                break;
                
            case 'clear':
                clearTerminal();
                break;
                
            case 'export':
                await exportProjectToTerminal();
                break;
                
            case 'cells':
                showActiveCells();
                break;
                
            case 'validate':
                if (args.length > 0) {
                    await validateFormulaInTerminal(args.join(' '));
                } else {
                    addTerminalLine('Usage: validate <formula>', 'terminal-error');
                }
                break;
                
            default:
                addTerminalLine(`Unknown command: ${cmd}. Type "help" for available commands.`, 'terminal-error');
        }
    } catch (error) {
        addTerminalLine(`Error executing command: ${error.message}`, 'terminal-error');
    }
}

function showTerminalHelp() {
    const helpText = [
        'Available commands:',
        '  help                 - Show this help message',
        '  status               - Show system status',
        '  benchmark [n]        - Run performance benchmark (default: 1000 iterations)',
        '  metrics              - Show performance metrics',
        '  cells                - List active cells',
        '  validate <formula>   - Validate a DVSL formula',
        '  export               - Export current project',
        '  clear                - Clear terminal',
        '',
        'Simulation controls:',
        '  Use the GUI panel for simulation start/stop/pause'
    ];
    
    helpText.forEach(line => addTerminalLine(line, 'terminal-info'));
}

async function showSystemStatus() {
    try {
        const status = await daseEngine.getSimulationStatus();
        addTerminalLine('System Status:', 'terminal-info');
        addTerminalLine(`  Engine initialized: ${status.initialized}`, 'terminal-info');
        addTerminalLine(`  Simulation running: ${status.running}`, 'terminal-info');
        addTerminalLine(`  Active cells: ${status.active_cells}`, 'terminal-info');
        
        if (status.metrics) {
            addTerminalLine(`  Operations/sec: ${Math.round(status.metrics.python_metrics?.operations_per_second || 0)}`, 'terminal-info');
        }
    } catch (error) {
        addTerminalLine(`Error getting status: ${error.message}`, 'terminal-error');
    }
}

async function runTerminalBenchmark(iterations) {
    addTerminalLine(`Starting benchmark with ${iterations} iterations...`, 'terminal-info');
    
    try {
        const results = await daseEngine.runBenchmark(iterations);
        if (results) {
            addTerminalLine('Benchmark completed:', 'terminal-success');
            if (results.cpp_metrics) {
                addTerminalLine(`  Operations: ${results.cpp_metrics.total_operations}`, 'terminal-info');
                addTerminalLine(`  ns/op: ${results.cpp_metrics.current_ns_per_op.toFixed(2)}`, 'terminal-info');
                addTerminalLine(`  ops/sec: ${Math.round(results.cpp_metrics.current_ops_per_second)}`, 'terminal-info');
                addTerminalLine(`  Speedup: ${results.cpp_metrics.speedup_factor.toFixed(2)}x`, 'terminal-info');
            }
        } else {
            addTerminalLine('Benchmark failed', 'terminal-error');
        }
    } catch (error) {
        addTerminalLine(`Benchmark error: ${error.message}`, 'terminal-error');
    }
}

async function showPerformanceMetrics() {
    try {
        const metrics = await daseEngine.getMetrics();
        addTerminalLine('Performance Metrics:', 'terminal-info');
        
        if (metrics.cpp_metrics) {
            addTerminalLine('C++ Engine:', 'terminal-info');
            addTerminalLine(`  Total operations: ${metrics.cpp_metrics.total_operations}`, 'terminal-info');
            addTerminalLine(`  AVX2 operations: ${metrics.cpp_metrics.avx2_operations}`, 'terminal-info');
            addTerminalLine(`  Current ns/op: ${metrics.cpp_metrics.current_ns_per_op.toFixed(2)}`, 'terminal-info');
            addTerminalLine(`  Current ops/sec: ${Math.round(metrics.cpp_metrics.current_ops_per_second)}`, 'terminal-info');
        }
        
        if (metrics.python_metrics) {
            addTerminalLine('Python Interface:', 'terminal-info');
            addTerminalLine(`  Operations/sec: ${Math.round(metrics.python_metrics.operations_per_second)}`, 'terminal-info');
            addTerminalLine(`  Total operations: ${metrics.python_metrics.total_operations}`, 'terminal-info');
        }
    } catch (error) {
        addTerminalLine(`Error getting metrics: ${error.message}`, 'terminal-error');
    }
}

function clearTerminal() {
    const terminalContent = document.getElementById('terminalContent');
    if (!terminalContent) return;
    
    // Keep only the input line
    const inputLine = terminalContent.querySelector('.terminal-input-line');
    terminalContent.innerHTML = '';
    if (inputLine) {
        terminalContent.appendChild(inputLine);
    }
}

async function exportProjectToTerminal() {
    try {
        const projectData = await daseEngine.exportProject();
        if (projectData) {
            const jsonStr = JSON.stringify(projectData, null, 2);
            addTerminalLine('Project exported:', 'terminal-success');
            addTerminalLine(`Data size: ${jsonStr.length} characters`, 'terminal-info');
            
            // Offer to download
            const blob = new Blob([jsonStr], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'dase_project.json';
            a.click();
            URL.revokeObjectURL(url);
            
            addTerminalLine('Download started: dase_project.json', 'terminal-success');
        }
    } catch (error) {
        addTerminalLine(`Export error: ${error.message}`, 'terminal-error');
    }
}

function showActiveCells() {
    const active = getActiveCells();
    addTerminalLine(`Active cells (${active.length}):`, 'terminal-info');
    
    active.forEach(cellId => {
        const cell = cellData[cellId];
        const type = cell.symbol ? 'symbol' : (cell.formula ? 'formula' : 'value');
        const content = cell.formula || cell.value || '';
        addTerminalLine(`  ${cellId}: ${type} - ${content.substring(0, 50)}`, 'terminal-info');
    });
}

async function validateFormulaInTerminal(formula) {
    try {
        const result = await daseEngine.validateFormula(formula);
        if (result.valid) {
            addTerminalLine(`Formula is valid: ${formula}`, 'terminal-success');
            if (result.parsed) {
                addTerminalLine(`Parsed as: ${result.parsed.type}`, 'terminal-info');
            }
        } else {
            addTerminalLine(`Formula is invalid: ${result.error}`, 'terminal-error');
        }
    } catch (error) {
        addTerminalLine(`Validation error: ${error.message}`, 'terminal-error');
    }
}

// Initialize engine when page loads
document.addEventListener('DOMContentLoaded', async () => {
    // Create global engine instance
    window.daseEngine = new EnhancedDVSLEngine();
    
    // Try to initialize
    const initialized = await window.daseEngine.initialize();
    if (initialized) {
        addTerminalLine && addTerminalLine('DASE engine connected successfully', 'terminal-success');
    } else {
        addTerminalLine && addTerminalLine('Warning: DASE engine connection failed. Running in simulation mode.', 'terminal-warning');
    }
});
