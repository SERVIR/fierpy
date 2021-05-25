import setuptools
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(name='fierpy',
    version='0.0.4',
    description='Python implementation of the Forecasting Inundation Extents using REOF method',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='http://github.com/servir/fierpy',
    packages=setuptools.find_packages(),
    author='Kel Markert',
    author_email='kel.markert@gmail.com',
    license='MIT',
    zip_safe=False,
    include_package_data=True,
    install_requires=[
        'numpy',
        'pandas',
        'xarray',
        'scikit-learn>=0.24',
        'geoglows',
        'eofs',
        'netcdf4',
    ],
    extras_require = {
        'opensarlab':  [
            'kernda',
            'jupyter',

        ]
    }
)
