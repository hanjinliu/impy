import sys

sys.stderr.write(
    """
    ===============================================================
    impy does not support `python setup.py install`. Please use

        $ python -m pip install .

    instead.
    ===============================================================
    """
)
sys.exit(1)

setup(
    name="impy-array",
    author="Hanjin Liu",
    author_email="liuhanjin-sc@g.ecc.u-tokyo.ac.jp",
    long_description_content_type="text/markdown",
    license="BSD 3-Clause",
    download_url="https://github.com/hanjinliu/impy",
    install_requires=[
        "numpy>=1.22",
        "scikit-image>=0.20.0",
        "pandas>=1.3",
        "dask>=2021.6.0",
        "napari>=0.4.17",
    ],
    python_requires=">=3.8",
    entry_points={"console_scripts": ["impy=impy.__main__:main",],},
    extras_require={
        "all": ["mrcfile", "zarr"],
        "zarr": ["zarr"],
        "mrc": ["mrcfile"],
    },
)
