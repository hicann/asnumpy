import os
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

# Custom build class integrating CMake build process
class CMakeBuild(build_ext):
    def run(self):
        # Get the project root directory path
        project_root = os.path.abspath(os.path.dirname(__file__))
        # Set the build temporary directory
        build_directory = os.path.join(project_root, 'build')
        if not os.path.exists(build_directory):
            os.makedirs(build_directory)

        # Set build type to Release
        cfg = 'Release'
        # Build CMake arguments list
        cmake_args = [
            f'-DCMAKE_BUILD_TYPE={cfg}',  # Build type
            f'-DPYTHON_EXECUTABLE={sys.executable}',  # Specify Python interpreter path
        ]

        # Build arguments configuration
        build_args = ['--config', cfg]
        cmake_args += ['-G', 'Ninja']  # Use Ninja build system
        # Parallel build arguments, determined by the number of CPU cores
        build_args += ['--', '-j', str(os.cpu_count() or 4)]

        # Execute CMake configure stage
        print(f"Running CMake configure in {build_directory}...")
        subprocess.check_call(['cmake', project_root] + cmake_args, cwd=build_directory)

        # Execute CMake build stage
        print(f"Building in {build_directory}...")
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=build_directory)

# Extension module definition
npu_array_extension = Extension(
    # Full module path: asnumpy.lib._NPUArray
    'asnumpy.lib.asnumpy_core',
    sources=[],  # No direct source files (built by CMake)
    # Specify library file search path
    library_dirs=['asnumpy/lib'],  # Build product storage path
    # Runtime library search path ($ORIGIN indicates the module directory)
    runtime_library_dirs=['$ORIGIN', '$ORIGIN/lib'],
    # Libraries to link (automatically matches libNPUArray.so and variants with version numbers)
    libraries=['asnumpy_core'],
    # Link arguments: Embed RPATH information into the binary
    extra_link_args=['-Wl,-rpath,$ORIGIN/lib']
)

# Package configuration
setup(
    name='asnumpy',  # Package name
    version='0.1.0',  # Version number
    # Included sub-packages
    packages=['asnumpy', 'asnumpy.lib'],
    # Package path mapping (empty string indicates the current directory)
    package_dir={'': '.'},
    # Package data file configuration (specifies binary files to include)
    package_data={'asnumpy.lib': ['*.so', '*.pyd']},
    include_package_data=True,  # Enable inclusion of package data files
    ext_modules=[npu_array_extension],  # Extension module list
    # Register build command class
    cmdclass={'build_ext': CMakeBuild},
    zip_safe=False,  # Disable zip installation (because it contains binary files)
)
