from setuptools import setup


setup(
    name='alnairjob',
    version='0.3',
    license='Apache-2.0 license',
    author="Centaurus Infrastructure",
    author_email='centaurusinfra@gmail.com',
    description="Alnair Job Dataset and DataLoader",
    package_dir={'': 'src'},
    py_modules=["AlnairJob"],             # Name of the python package
    python_requires='>=3.6',              # Minimum version requirement of the package
    url='https://github.com/CentaurusInfra/alnair/tree/main/storage-caching/k-v-store/src/pyalnair',
    keywords='Alnair',
    install_requires=[
          'torch',
          'redis',
          'pickle-mixin'
      ],
)