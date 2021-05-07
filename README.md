# Wrap LISFLOOD-FP 

A python wrap for the LISFLOOD-FP hydrodinamic numerical model.

## Table of contents
1. [Description](#desc)
2. [Main Contents](#mc)
3. [Documentation](#doc)
4. [Install](#ins)
    1. [Install from sources](#ins_src)
    2. [Install LISFLOOD-FP numerical model](#ins_lf)
5. [Examples](#exp)
    1. [.py examples](#exp_1)
    2. [notebook examples](#exp_2)
6. [Contributors](#ctr)
7. [License](#lic)



<a name="desc"></a>
## Description

LISFLOOD-FP is a hydrodinamic numerical model designed to simulate floodplain inundation events. 

This python toolbox includes a set of methodologies to prepare and run inundation cases, and also postprocess and plot the cases output. 

Coastal inundation cases are achieved by preprocessing the study site digital terrain model and defining coastal water inflow by shoreline transects.


<a name="mc"></a>
## Main contents

[wlisflood](./wlisflood/): LISFLOOD-FP numerical model toolbox 
- [io](./wlisflood/io.py): LISFLOOD-FP numerical model input/output operations
- [wrap](./wlisflood/wrap.py): LISFLOOD-FP numerical model python wrap 
- [plots](./wlisflood/plots.py): plotting module 


<a name="doc"></a>
## Documentation

LISFLOOD-FP numerical model official web site <http://www.bristol.ac.uk/geography/research/hydrology/models/lisflood/>

LISFLOOD-FP user manual (v5.9.6) can be found inside docs subfolder [here](./docs/lisflood-manual-v5.9.6.pdf)

Digital terrain models can be downloaded from SRTM 90m DEM Digital Elevation Database <https://srtm.csi.cgiar.org/srtmdata/>


<a name="ins"></a>
## Install
- - -

Source code is currently privately hosted on GitLab at:  <https://gitlab.com/geoocean/bluemath/hywaves/tree/master> 


<a name="ins_src"></a>
### Install from sources

This toolbox is developed using python 3.7, the use of a virtual environment is highly recommended.

Install requirements. Navigate to the base root of [wlisflood](./) and execute:

```bash
   pip install -r requirements/requirements.txt
```

Basemap library is used for some of the plots inside this project, to install it execute:

```bash
   pip install git+https://github.com/matplotlib/basemap.git
```


<a name="ins_lf"></a>
### Install LISFLOOD-FP 

Currently Linux and OS X precompiled LISFLOOD-FP executables are attached to the repository at the [bin](./wlisflood/resources/bin/) folder, no installation is needed


<a name="exp"></a>
## Examples:
- - -

<a name="exp_1"></a>
### .py examples 

- [demo 01 - Coastal flooding event](./scripts/demo_coastal_flooding.py): DEM preprocessing and coastal inundation with shoreline transects inflow methodology 
- [demo 02 - Rainfall flooding event](./scripts/demo_rainfall_flooding.py): Simple flooding event due to constant rainfall. No DEM preprocessing.

<a name="exp_2"></a>
### notebook examples 

- [notebook - TODO](./notebooks/): TODO


<a name="ctr"></a>
## Contributors:

Manuel Zornoza Aguado (manuel.zornoza@unican.es)\
Sara Ortega Van Vloten (sara.ortegav@unican.es)\
Nicolas Ripoll Cabarga (ripolln@unican.es)


<a name="lic"></a>
## License

This project is licensed under the MIT License - see the [license](./LICENSE.txt) file for details

