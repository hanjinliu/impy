from setuptools import setup, find_packages

with open("impy/__init__.py", encoding="utf-8") as f:
    line = next(iter(f))
    VERSION = line.strip().split()[-1][1:-1]
      
with open("README.md", "r") as f:
    readme = f.read()
    
setup(name="impy-array",
      version=VERSION,
      description="Speed up coding/extending image analysis in Python.",
      author="Hanjin Liu",
      author_email="liuhanjin-sc@g.ecc.u-tokyo.ac.jp",
      long_description=readme,
      long_description_content_type="text/markdown",
      license="BSD 3-Clause",
      download_url="https://github.com/hanjinliu/impy",
      packages=find_packages(exclude=["docs", "tests"]),
      install_requires=[
            "numpy>=1.21",
            "scikit-image>=0.18.2",
            "pandas>=1.3",
            "dask>=2021.6.0",
            "napari>=0.4.13",
            "qtpy>=1.10.0",
      ],
      python_requires=">=3.8",
      entry_points={"console_scripts": ["impy=impy.__main__:main",],}
      )