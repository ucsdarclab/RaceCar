from setuptools import setup, find_packages

setup(
    name='racecar',
    version='0.0.1',
    description='A gym environment for a RaceCar',
    long_description='A package that simulate the dynamics of a RaceCar',
    author='Michael Yip, Jacob Johnson',
    author_email='m1yip@ucsd.edu, jjj025@ucsd.edu',
    url='',
    download_url='',
    license='BSD License 2.0',
    install_requires=[
        'numpy>=1.17.1',
        'matplotlib>=3.1.1'
    ],
    package_data={'': ['input']},
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha', 'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    packages=find_packages(),
)
