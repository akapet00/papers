# Human Tissues Dielectric Properties

## Overview

The dielectric properties of the tissue (conductivity [S/m] and relative permitivity) w.r.t. frequency in GHz spectrum. 

Dielectric property values are based on the work of C. Gabriel at el. in [1], where a parametric model was developed to describe the variation of dielectric properties of (human) tissues as a function of frequency.
The spectrum from 10 Hz to 100 GHz is modeled to 4 dispersion regions with the frequency dependence within each region expressed as a Cole-Cole term.

## Web

An Internet resource for the calculation of the dielectric properties of human body tissues in the frequency range 10 Hz - 100 GHz can be found [here](http://niremf.ifac.cnr.it/tissprop/). Official data generator is available [here](http://niremf.ifac.cnr.it/tissprop/htmlclie/htmlclie.php). 

## Locally available data

Supported tissue types and their respective properties are available in this directory and are stored into `tissue_dielectric_properties.csv` file.
Frequency range supported locally is [10, 100] GHz.
This file contains six colums, where 1st column is the type of human tissue, 2nd columnd is frequency in Hz, 3rd column is the tissue conductivity in S/m, 4th column is the relative permitivity, 5th columns is the loss tangent and finally, 6th column is the penetration depth into respective tissue, which is defined as the distance beneath the surface at which the specific power absorption rate has fallen to a factor of $1/e$ below that at the surface of the tissue.

## Rights

See [2].

## References

[1] S. Gabriel, R. W. Lau and C. Gabriel: "The dielectric properties of biological tissues: III. Parametric models for the dielectric spectrum of tissues", Phys. Med. Biol. 41 (1996), 2271-2293.

[2] D.Andreuccetti, R.Fossi and C.Petrucci: "An Internet resource for the calculation of the dielectric properties of body tissues in the frequency range 10 Hz - 100 GHz". IFAC-CNR, Florence (Italy), 1997. Based on data published by C. Gabriel et al. in 1996.