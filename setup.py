from setuptools import setup, find_packages

with open("impy/__init__.py", encoding="utf-8") as f:
    line = next(iter(f))
    VERSION = line.strip().split()[-1][1:-1]
      
with open("README.md", "r") as f:
    readme = f.read()
    
setup(name="impy-array",
      version=VERSION,
      description="Speed up image analysis in Python with efficient reading, batch-processing, " \
                  "viewing functions and easily extend your own function for batch processing.",
      author="Hanjin Liu",
      author_email="liuhanjin-sc@g.ecc.u-tokyo.ac.jp",
      long_description=readme,
      long_description_content_type="text/markdown",
      license="BSD 3-Clause",
      download_url="https://github.com/hanjinliu/impy",
      packages=find_packages(exclude=["docs"]),
      install_requires=[
            "scikit-image>=0.18.2",
            "pandas>=1.3",
            "dask>=2021.6.0",
            "tifffile>=2021.6.14",
            "napari>=0.4.12",
            "superqt>=0.2.4",
            "qtpy>=1.10.0",
      ],
      python_requires=">=3.7",
      )