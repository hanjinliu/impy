from setuptools import setup, find_packages
from .impy import __version__

setup(name="impy",
      version=__version__,
      description="Numpy and scikit-image based image analysis tool",
      author="Hanjin Liu",
      author_email="liuhanjin-sc@g.ecc.u-tokyo.ac.jp",
      license="GPLv2",
      packages=find_packages(),
      install_requires=[
            "scikit-image",
            "numpy",
            "matplotlib",
      ],
      python_requires=">=3.6",
      )