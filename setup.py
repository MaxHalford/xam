from setuptools import find_packages
from setuptools import setup

setup(
    author='Max Halford',
    author_email='maxhalford25@gmail.com',
    description="Max Halford's personal data science toolkit",
    license='MIT',
    name='xam',
    python_requires='>=3.5.0',
    install_requires=[
        'lightgbm>=2.1.2',
        'matplotlib>=2.2.2',
        'numpy>=1.14.0',
        'pandas>=0.22.0',
        'scipy>=1.0.1',
        'scikit-learn>=0.20.0'
    ],
    packages=find_packages(exclude=['examples']),
    url='https://github.com/MaxHalford/xam',
    version='0.0.1dev',
)
