from setuptools import setup, find_packages

setup(
<<<<<<< Updated upstream
    name="geometric_clustering",
    version="1.0",
    install_requires=[
        "numpy>=1.19.5",
        "scipy>=1.6.0",
        "networkx>=2.5",
        "matplotlib>=3.3.3",
        "tqdm>=4.56.0",
        "POT>=0.7.0",
        "pygenstability>=0.0.2",
    ],
    packages=find_packages(),
)
=======
        name = 'geocluster',
        version = '1.0',
        install_requires=['numpy', 
                          'scipy', 
                          'networkx', 
                          'matplotlib', 
                          'pyyaml',
                          'tqdm',
                          'sklearn',
                          'POT'],
        packages = find_packages(),                       
      )
>>>>>>> Stashed changes
