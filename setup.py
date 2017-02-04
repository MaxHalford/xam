from distutils.core import find_packages
from distutils.core import setup

setup(
    author='Max Halford',
    author_email='maxhalford25@gmail.com',
    description='Data science utilities library',
    license='MIT',
    name='xam',
    packages=find_packages(exclude=['tests']),
    url='https://github.com/MaxHalford/xam',
    version='0.0.1dev',
)
