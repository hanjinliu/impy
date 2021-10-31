from setuptools import setup, find_packages

with open("impy/__init__.py", encoding="utf-8") as f:
    for line in f:
        if (line.startswith("__version__")):
            VERSION = line.strip().split()[-1][1:-1]
            break
      
setup(name="impy",
      version=VERSION,
      description="Speed up image analysis in Python with efficient reading, batch-processing, " \
                  "viewing functions and easily extend your own function for batch processing.",
      author="Hanjin Liu",
      author_email="liuhanjin-sc@g.ecc.u-tokyo.ac.jp",
      license="BSD 3-Clause",
      packages=find_packages(),
      install_requires=[
            "scikit-image>=0.18",
            "numpy>=1.17",
            "scipy>=1.6.3",
            "matplotlib",
            "pandas>=1",
            "dask>=2021.6.0",
            "tifffile>=2021.6.14",
            "magicgui>=0.3.2",
            "napari>=0.4.12",
            "qtpy>=1.10.0",
      ],
      python_requires=">=3.7",
      )