# claustrum_imaging_manuscript

This is a repository containing code to generate figures for the claustrum imaging manuscript

## Setup

Running the notebooks in this repo in a dedicated environment is strongly encouraged. For example, if using Conda, create a new Python 3.7 environment as follows:

    conda create -n claustrum_imaging_manuscript python=3.7

Activate the environment:

    conda activate claustrum_imaging_manuscript

Then, to make the environment visible in Jupyter:

    python -m ipykernel install --user --name claustrum_imaging_manuscript

To install dependencies:

    pip install .

or

    python setup.py

Follow instructions in the `/data` folder readme to download necessary data (data not included in the github repository)

## Use

Each `figure_X` folder contains a Jupyter Notebook with code to generate the given figure. Notebooks should be executed in order, or by selecting `cell > Run All`

## Level of Support

We are not currently supporting this code, but simply releasing it to the community AS IS and are not able to provide any guarantees of support. The community is welcome to submit issues, but you should not expect an active response.
