# parona

Scripts for data reduction of observations of mainly X-ray facilities.

Two Python classes to reduce XMM-Newton and NuSTAR data. These classes make use of SAOimageDS9 to select regions on the images somewhat interactively. The regions are then used to extract spectra and light curves. 

### Installation

First, install Heasoft and XMM-Newton SAS (see [here](https://heasarc.gsfc.nasa.gov/docs/software/lheasoft/) and [here](https://www.cosmos.esa.int/web/xmm-newton/sas-installation)). Then install the package with pip. It is recommended to install the package in the same Python environment as Heasoft and SAS.

```bash
pip install git+https://github.com/mlefkir/parona.git
```
