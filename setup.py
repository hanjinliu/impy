from setuptools import setup, find_packages

with open("impy/__init__.py", encoding="utf-8") as f:
    for line in f:
        if (line.startswith("__version__")):
            VERSION = line.strip().split()[-1][1:-1]
            break
      
setup(name="impy",
      version=VERSION,
      description="Numpy and scikit-image based image analysis tool",
      author="Hanjin Liu",
      author_email="liuhanjin-sc@g.ecc.u-tokyo.ac.jp",
      license="GPLv2",
      packages=find_packages(),
      install_requires=[
            "scikit-image>=0.18",
            "numpy>=1.17",
            "matplotlib",
            "pandas>=0",
      ],
      python_requires=">=3.7",
      )