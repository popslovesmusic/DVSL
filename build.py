#!/usr/bin/env python3
"""
DASE Build Script
Compiles C++ engine and sets up Python environment
"""

import os
import sys
import subprocess
import platform
import shutil
from pathlib import Path

def check_dependencies():
    """Check for required build dependencies"""
    print("Checking build dependencies...")
    
    dependencies = {
        'cmake': 'cmake --version',
        'gcc/g++': 'g++ --version' if platform.system() != 'Windows' else 'cl',
        'python': 'python --version'
    }
    
    missing = []
    for name, cmd in dependencies.items():
        try:
            subprocess.run(cmd.split(), capture_output=True, check=True)
            print(f"✓ {name} found")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print(f"✗ {name} not found")
            missing.append(name)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        return False
    
    return True

def install_python_dependencies():
    """Install required Python packages"""
    print("\nInstalling Python dependencies...")
    
    requirements = [
        'pybind11[global]',
        'numpy>=1.19.0',
        'scipy>=1.5.0',
        'flask>=2.0.0',
        'flask-cors>=3.0.0',
    ]
    
    for req in requirements:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', req], check=True)
            print(f"✓ Installed {req}")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {req}")
            return False
    
    return True

def compile_cpp_engine():
    """Compile the C++ engine with Python bindings"""
    print("\nCompiling C++ engine...")
    
    # Check if files exist
    required_files = [
        'analog_universal_node_engine_avx2.cpp',
        'analog_universal_node_engine_avx2.h',
        'python_bindings.cpp',
        'setup.py'
    ]
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    if missing_files:
        print(f"Missing required files: {', '.join(missing_files)}")
        return False
    
    try:
        # Clean previous builds
        build_dirs = ['build', 'dist', 'dase_engine.egg-info']
        for build_dir in build_dirs:
            if os.path.exists(build_dir):
                shutil.rmtree(build_dir)
                print(f"Cleaned {build_dir}")
        
        # Build the extension
        cmd = [sys.executable, 'setup.py', 'build_ext', '--inplace']
        subprocess.run(cmd, check=True)
        print("✓ C++ engine compiled successfully")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ Compilation failed: {e}")
        return False

def test_engine():
    """Test the compiled engine"""
    print("\nTesting compiled engine...")
    
    try:
        # Test import
        import dase_engine
        print("✓ Engine import successful")
        
        # Test basic functionality
        if hasattr(dase_engine, 'CPUFeatures'):
            avx2_support = dase_engine.CPUFeatures.has_avx2()
            fma_support = dase_engine.CPUFeatures.has_fma()
            print(f"✓ AVX2 support: {avx2_support}")
            print(f"✓ FMA support: {fma_support}")
        
        # Test engine creation
        engine = dase_engine.AnalogCellularEngine(100)
        print("✓ Engine creation successful")
        
        # Test basic operation
        result = engine.process_signal_wave(1.0, 0.5)
        print(f"✓ Basic operation test: {result}")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Engine test failed: {e}")
        return False

def create_launch_script():
    """Create a launch script for the web server"""
    print("\nCreating launch script...")
    
    script_content = '''#!/usr/bin/env python3
"""
DASE Launcher
Launch the DASE web server with proper environment
"""

import os
import sys
import webbrowser
import time
from pathlib import Path

# Add current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from web_server import run_server
    print("Starting DASE Web Server...")
    print("Interface will be available at: http://localhost:5000")
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(2)
        webbrowser.open('http://localhost:5000')
    
    import threading
    browser_thread = threading.Thread(target=open_browser, daemon=True)
    browser_thread.start()
    
    # Run the server
    run_server(host='localhost', port=5000, debug=False)
    
except ImportError as e:
    print(f"Error: Missing dependencies - {e}")
    print("Please run: python build.py")
    sys.exit(1)
except KeyboardInterrupt:
    print("\\nShutting down server...")
except Exception as e:
    print(f"Error starting server: {e}")
    sys.exit(1)
'''
    
    with open('launch_dase.py', 'w') as f:
        f.write(script_content)
    
    # Make executable on Unix systems
    if platform.system() != 'Windows':
        os.chmod('launch_dase.py', 0o755)
    
    print("✓ Launch script created: launch_dase.py")

def create_requirements_file():
    """Create requirements.txt file"""
    requirements_content = '''# DASE Requirements
pybind11[global]>=2.6.0
numpy>=1.19.0
scipy>=1.5.0
flask>=2.0.0
flask-cors>=3.0.0

# Optional dependencies for enhanced functionality
matplotlib>=3.3.0
plotly>=5.0.0
websockets>=9.0.0
'''
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)
    
    print("✓ Requirements file created: requirements.txt")

def main():
    """Main build process"""
    print("DASE Build System")
    print("=" * 50)
    
    # Check platform
    print(f"Platform: {platform.system()} {platform.machine()}")
    print(f"Python: {sys.version}")
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        sys.exit(1)
    
    # Step 2: Install Python dependencies
    if not install_python_dependencies():
        print("\nFailed to install Python dependencies.")
        sys.exit(1)
    
    # Step 3: Compile C++ engine
    if not compile_cpp_engine():
        print("\nFailed to compile C++ engine.")
        print("Check error messages above for details.")
        sys.exit(1)
    
    # Step 4: Test the engine
    if not test_engine():
        print("\nEngine test failed.")
        sys.exit(1)
    
    # Step 5: Create launch scripts and requirements
    create_launch_script()
    create_requirements_file()
    
    print("\n" + "=" * 50)
    print("✓ Build completed successfully!")
    print("\nTo start DASE:")
    print("  python launch_dase.py")
    print("\nOr manually:")
    print("  python web_server.py")
    print("\nThen open: http://localhost:5000")

if __name__ == '__main__':
    main()
