from setuptools import setup, find_packages

setup(name="impy",
      version="1.1.2",
      description="Numpy based image analysis tool",
      author="Hanjin Liu",
      author_email="liuhanjin-sc@g.ecc.u-tokyo.ac.jp",
      license="GPLv2",
      packages=find_packages(),
      install_requires=[
            "scikit-image",
            "numpy",
            "matplotlib",
            "ipywidgets",
      ],
      python_requires=">=3.6",
      )