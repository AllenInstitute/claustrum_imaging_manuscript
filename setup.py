import setuptools

setuptools.setup(
    name="claustrum_imaging_manuscript",
    version="1.0.0",
    author="Douglas R Ollerenshaw",
    author_email="dougo@alleninstitute.org",
    description="figures for claustrum imaging manuscript",
    install_requires=[
        "numpy==1.20.1",
        "pandas==1.2.2",
        "scipy==1.5.3",
        "matplotlib==3.3.2",
        "seaborn==0.11.0",
    ],
    classifiers=[
        'Programming Language :: Python :: 3.7',
    ],
)
