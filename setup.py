import os
import sys
from numpy.distutils.misc_util import Configuration
from numpy.distutils.core import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def configuration(parent_package='',top_path=None):
    
    config = Configuration('pamtraRadarSimulator', parent_package, top_path,
        version = '0.1',
        author  = "Maximilin Maahn",
        author_email = "maximilian.maahn@colorado.edu",
        description = "Simulate a radar Doppler spectrum",
        license = "GPL v3",
        python_requires='>=3.5',
        url = 'https://github.com/maahn/pamtraRadarSimulator',
        download_url = 'https://github.com/maahn/pamtraRadarSimulator/releases/download/0.1/pamtraRadarSimulator-0.1.zip',
        long_description = read('README.rst'),
        classifiers = [
            "Development Status :: 3 - Alpha",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Fortran",
            "Programming Language :: Python",
            "Intended Audience :: Science/Research",
            "Topic :: Scientific/Engineering :: Atmospheric Science",
            ]
    )


    kw = {}
    if sys.platform == 'darwin':
        kw['extra_link_args'] = ['-undefined dynamic_lookup', '-bundle']
    config.add_extension('pamtraRadarSimulatorLib',
        sources=[
            'pamtraRadarSimulator/pamtraRadarSimulatorLib/pamtraRadarSimulatorLib.pyf',
            'pamtraRadarSimulator/pamtraRadarSimulatorLib/kinds.f90',
            'pamtraRadarSimulator/pamtraRadarSimulatorLib/report_module.f90',
            'pamtraRadarSimulator/pamtraRadarSimulatorLib/constants.f90',
            'pamtraRadarSimulator/pamtraRadarSimulatorLib/dsort.f90',
            'pamtraRadarSimulator/pamtraRadarSimulatorLib/convolution.f90',
            'pamtraRadarSimulator/pamtraRadarSimulatorLib/viscosity_air.f90',
            'pamtraRadarSimulator/pamtraRadarSimulatorLib/random.f90',
            'pamtraRadarSimulator/pamtraRadarSimulatorLib/rescale_spectra.f90',
            'pamtraRadarSimulator/pamtraRadarSimulatorLib/dia2vel.f90',
            'pamtraRadarSimulator/pamtraRadarSimulatorLib/radar_simulator.f90',
            'pamtraRadarSimulator/pamtraRadarSimulatorLib/radar_spectral_broadening.f90',
            'pamtraRadarSimulator/pamtraRadarSimulatorLib/radar_spectrum.f90',
            'pamtraRadarSimulator/pamtraRadarSimulatorLib/rho_air.f90',
                 ],
        library_dirs = ['../dfftpack/'],
        libraries = ['dfftpack','lapack'],
        **kw)

    return config


if __name__ == "__main__":

    
    setup(configuration=configuration,
        packages = ['pamtraRadarSimulator','pamtraRadarSimulator.pamtraRadarSimulatorLib'],        
        # package_data = {
        #     'pamtraRadarSimulator': ['file'],
        # },
        platforms = ['any'],
        requires = ['numpy', 'scipy'])

