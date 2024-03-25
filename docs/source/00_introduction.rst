=================
Introduction
=================

In this section we provide an introduction to satellite missions, sensors and products available with GPM-API.
Concepts are introduced in a progressive manner, but feel free to skip to the section that is of interest to you.

.. contents:: Table of Contents
   :depth: 2
   :local:


.. _gpm_mission:

GPM Mission
---------------

The Global Precipitation Measurement (GPM) mission, an international collaborative effort between NASA, JAXA, and other space agencies,
was initiated following the success of the `Tropical Rainfall Measuring Mission (TRMM) satellite <https://gpm.nasa.gov/missions/trmm>`_ launched in November 1997.

The GPM mission aims to acquire accurate and frequent global observations of Earth's precipitation.
The mission relies on the TRMM, the `GPM Core Observatory satellite <https://gpm.nasa.gov/missions/GPM/core-observatory>`_ and a constellation of satellites equipped
with microwave radiometers which forms the so-called `GPM Constellation <https://gpm.nasa.gov/missions/GPM/constellation>`_.

The reference sensors for the GPM mission are the:

- TRMM Precipitation Radar (PR)
- TRMM Microwave Imager (TMI)
- GPM Dual-frequency Precipitation Radar (DPR)
- GPM Microwave Imager (GMI)

The GPM Data Archive currently includes satellite data records that extend back to 1987.
This extensive archive is the result of contributions from two spaceborne radars and a fleet of 35 passive microwave (PMW) sensors that forms the so-called GPM constellation.

The data are organized into various product levels, encompassing raw and calibrated observations (Level 1), intermediate geophysical retrieval products (Level 2),
and spatio-temporally integrated datasets from single and multiple satellites (Level 3).

While the GPM mission is renowned for providing a long-term data records of precipitation, its satellite data have also been essential for the quantification,
monitoring and understanding of a broad spectrum of atmospheric, ocean and terrestrial surface processes.
Examples include monitoring sea-ice concentration and snow-cover extent, estimating ocean wind speeds and sea surface temperatures,
and profiling atmospheric humidity and temperature.
Moreover, GPM data have also been crucial for identifying global hotspots for hail and intense thunderstorms, analyzing storm structures
and examining the latent heat release that drives the atmospheric circulation.


.. _trmm_satellite:

TRMM satellite
---------------

The TRMM satellite, launched in November 1997 as a collaborative effort between the
United States' National Aeronautics and Space Administration (NASA) and the
Japan Aerospace Exploration Agency (JAXA),
marked a significant milestone for the satellite remote sensing observation of tropical precipitations between 37°N/S.
Its primary objective was to accurately measure rainfall associated with tropical convective activity, which play a crucial role in the global dynamics of the atmospheric circulation.

Equipped with pioneering technology including the first spaceborne
`Precipitation Radar (PR) <https://gpm.nasa.gov/missions/TRMM/satellite/PR>`_,
the `TRMM Microwave Imager (TMI) <https://gpm.nasa.gov/missions/TRMM/satellite/TMI>`_,
the `Visible and Infrared Scanner (VIRS) imager <https://gpm.nasa.gov/missions/TRMM/satellite/virs>`_,
and the `Lightning Imaging Sensor (LIS) <https://ghrc.nsstc.nasa.gov/lightning/overview_lis_instrument.html>`_,
TRMM aimed to revolutionize rainfall observation.

The combined use of PR and TMI significantly enhanced rainfall estimation accuracy over the tropics and subtropics.
Moreover, PR provided unprecedented insights into the three-dimensional structure of cyclones over the ocean,
as well as rainfall characteristics of the Madden-Julian Oscillation and other climate phenomena such as El Niño and La Niña.

The success of the TRMM mission underscored the potential of satellite remote sensing in advancing our understanding of Earth's water cycle
and improving weather forecasting capabilities.

.. _trmm_pr:

TRMM PR
~~~~~~~~

The TRMM `Precipitation Radar (PR) <https://gpm.nasa.gov/missions/TRMM/satellite/PR>`_ is a Ku-band (13.6 GHz) cross-track scanning radar with a swath coverage of 215 km.
The PR accurately quantifies heavier rainfall in the tropics and the subtropics, but lacks of sensitivity to detect light precipitation and drizzle.

If you plan to use PR measurements, it's essential to consider the satellite orbit boost (from 350 km to 402.5 km) occured in August 2011 to extend the satellite lifetime.
Following the boost, the swath coverage has widened to 220 km but the spatial resolution of PR footprints has increased from 4.3 km to 5 km
and the PR sensitivity has reduced by more than 1 dBZ.


.. _trmmi_tmi:

TRMM TMI
~~~~~~~~

The TRMM Microwave Imager (TMI) is a passive microwave, dual-polarization conically scanning radiometer equipped with 9 channels.
ranging from 10.7 GHz to 85.5 GHz (the 21.3 GHz has only single polarization).
The TMI has a swath coverage of 760 km and a spatial resolution ranging between 5 and 37 km at 85.5 and 10.65 GHz respectively.

Build upon the highly successful Special Sensor Microwave/Imager (SSM/I) sensors onboard of the Defense Meteorological Satellite Program (DMSP) satellites, TMI has
an additional 10.7 GHz channel and an improved spatial resolution (due to the lower flying altitude).

If you plan to use TMI measurements, it's essential to consider the satellite orbit boost (from 350 km to 402.5 km) occured in August 2011 to extend the satellite lifetime.
Following the boost, the swath coverage has widened and the instantanoues field of fiew of each channel has enlarged.


.. _gpm_core_satellite:

GPM-Core Observatory
---------------------

The GPM Core Observatory satellite, a joint collaboration between NASA and JAXA, was launched on February 28, 2014,
building upon the legacy of the TRMM to extend precipitation measurement capabilities from the tropics to higher latitudes (68°N/S).

Equipped with advanced instruments such as the
`Dual-frequency Precipitation Radar (DPR) <https://gpm.nasa.gov/missions/GPM/DPR>`_ and the
`GPM Microwave Imager (GMI) <https://gpm.nasa.gov/missions/GPM/GMI>`_,
the GPM Core Observatory can accurately measure a wide range of precipitation types, from light rain and snowfall to heavy tropical rainstorm.

Operating in a non-sun-synchronous orbit at a 65° inclination, the GPM Core Observatory is strategically positioned
to sample the diurnal cycle of precipitation over a wide geographic area, a capability not shared by other GPM constellation polar-orbiting sensors,
which acquire observations at fixed local times.

Moreover, this orbit enable to obtain coincident measurements with other PMW sensors within the GPM Constellation.
Consequently, this allow for the use of the GMI as a common radiometric reference standard for intersensor calibration
across the full range of microwave frequencies present in the GPM Constellation microwave radiometers.
This calibration process enhances the consistency and quality of derived precipitation estimates.

The video here below provides an nice overview of the GPM Core Observatory satellite.
..  youtube:: eM78gFFxAII
   :align: center


.. _gpm_dpr:

GPM DPR
~~~~~~~~

The GPM Dual-frequency Precipitation Radar (DPR) features a Ka-band precipitation radar (KaPR)
operating at 35.5 GHz and a Ku-band precipitation radar (KuPR) at 13.6 GHz.

While the KaPR instrument has been designed to detect weak rainfall and snowfall beyond the KuPR's sensitivity,
the KuPR excels at quantifying heavier rainfall and extends the long-term record of TRMM PR in the tropics and the subtropics.

Together, KaPR and KuPR enable three-dimensional dual-frequency observations of precipitation structures and accurate measurement across precipitation types.
This capability spans from heavy rainfall in the tropics to weak rainfall in mid-to-high latitudes and snowfall in high-latitudes.

.. figure:: https://www.eorc.jaxa.jp/GPM/image/overview-dpr.png
   :alt: GPM DPR Overview
   :align: center

   GPM DPR Overview

Note, however, that despite its detailed insights into precipitation structure and dynamics,
the DPR exhibits low sensitivity to light precipitation and drizzle, resulting in significant portions of the lightest precipitation going undetected.

Nonetheless, its detection capabilities miss only a small fraction of the total rain volume due to the relatively minor contribution from light rain and snowfall.
Comparison with CloudSat measurements revelead that DPR does not detect more than 90% of the snowfall identified by CloudSat and tends to underestimate surface snowfall accumulation.
It is worth noting, however, that recent studies have shown advancements in reducing clutter contamination and improving receiver noise-reducing algorithms, which have led to improved detection of light precipitation.

If you plan to analyze or utilize DPR measurements, it's essential to consider that the scan pattern of KaPR changed on May 21, 2018.

Prior to the scan pattern change, the dual-frequency information was only available within a narrow inner swath of 125 km.
However, after the pattern change, dual-frequency observations are available across the full swath of approximately 245 km.

The figure and video here below illustrate the scan pattern change.

.. figure:: https://www.eorc.jaxa.jp/GPM/en/image/scanpt_Fig2_en.png
   :alt: GPM DPR Scan Pattern Change
   :align: center


.. raw:: html

   <div style="display: flex;">
     <div style="flex: 50%; padding: 10px;">
       <iframe width="100%" height="315" src="https://www.youtube.com/embed/5voFOWbZtTs" frameborder="0" allowfullscreen></iframe>
       <p>Scan Pattern Before May 21, 2018</p>
     </div>
     <div style="flex: 50%; padding: 10px;">
       <iframe width="100%" height="315" src="https://www.youtube.com/embed/dTdMeX1RNEw" frameborder="0" allowfullscreen></iframe>
       <p>Scan Pattern After May 21, 2018</p>
     </div>
   </div>


Please also note that the GPM-CO satellite underwent an orbit boost (from 407 km to 442 km) the 7/8 November 2023 to extend the satellite lifetime.
Following the boost, the swath coverage has widened to 268 km but the spatial resolution of DPR footprints has increased from 5 km to 5.48 km
and the DPR sensitivity has reduced by approximately 0.8-0.9 dBZ.


.. _gpm_gmi:

GPM GMI
~~~~~~~~

The GPM Microwave Imager (GMI) is a dual-polarization, multi-channel, conical- scanning, microwave radiometer
equipped with 13 channels ranging from 10 GHz to 183 GHz.

Compared to the TMI instrument on TRMM, the GMI has an improved spatial resolution and carries 4 additional high-frequency channels near 166 GHz and 183 GHz.

Each channel is sensitive to a different frequency of microwave energy naturally emitted from or affected by precipitation.
Lower frequency channels are sensitive to heavy and moderate rainfall, while the high-frequency channels are affected by water vapor and ice particles.

A visual overview of the GMI channels is depicted in the figure below:

.. figure:: https://gpm.nasa.gov/sites/default/files/2019-11/GMI-13-channels.jpg
   :alt: GPM GMI channels
   :align: center

   Overview of the GPM GMI channels

It's important to highlight that the channel footprint size decreases with increasing frequency,
and the high-frequency channels are not aligned with the low-frequency channels.

However, the  ``1C-GMI-R`` product addresses this "problem" and provides the low-frequency channels collocated with the nearest neighbor high-frequency channels.

Furthermore, it's worth noting that only the central portion of the GMI swath overlaps the radar swath, with a delay of approximately 67 seconds caused by
the spacecraft motion and geometric considerations.

If you plan to use GMI measurements, it's essential to consider the satellite orbit boost (from 407 km to 442 km) the 7/8 November 2023 to extend the satellite lifetime.
Following the boost, the swath coverage has widened and the instantanoues field of fiew of each channel has enlarged.

The data acquired by GMI are used as a reference standard for intercalibrating the passive microwave sensors within the GPM Constellation,
ensuring consistency and accuracy in precipitation estimation across multiple satellite platforms.

.. _gpm_constellation:

GPM constellation
--------------------

The GPM constellation is composed by satellites of various space agencies , each equipped with microwave radiometers.
These radiometers operate across a range of frequencies, from 6 to 183 GHz, and include both conical-scanning and cross-track-scanning instruments.

Low-frequency channels are employed for rainfall estimation since the launch of the first SSM/I instrument in 1987.
Instead, high-frequency microwave channels were originally designed for water vapor profiling but resulted particularly useful
for discerning precipitation in regions with uncertain land surface emissivities, such as frozen and snow-covered areas.

Over the years, the composition of sensors within the constellation has evolved, leading to changes in spatial coverage and sampling frequency.
These changes are influenced by the number of operational sensors and their respective orbits.

The operational timeline of the GPM constellation is depicted in the figure below.

.. figure:: https://www.researchgate.net/profile/Daniel-Watters-3/publication/344906720/figure/fig1/AS:951371750719489@1603836109309/A-timeline-of-the-GPM-constellation-of-spaceborne-radars-and-passive-microwave.ppm
   :alt: GPM Constellation Timeline
   :align: center

It's important to note that not all existing PMW (Passive Microwave) sensors currently in orbit are part of the GPM constellation.
For instance, the constellation does not include 7 Chinese FY-3 Microwave Radiation Imagers (MWRI) and 6 Microwave Humidity Sounders (MWHS),
as well as 5 Russian Imaging/Sounding Microwave Radiometers (MTVZA).

The `WindSat Polarimetric Microwave Radiometer <https://www.eoportal.org/satellite-missions/coriolis#mission-status>`_ is also not part of the constellation.
Furthermore, recent satellite missions such as `TROPICS <https://weather.ndc.nasa.gov/tropics/>`_ and `TEMPEST <https://tempest.colostate.edu/>`_,
as well as private industry sensors from `Tomorrow.io <https://www.tomorrow.io/space/sounder/>`_ and `GEMS <https://weatherstream.com/gems/>`_,
are also yet not integrated into the GPM Constellation.

You can find additional reference to all the sensors in the :ref:`useful_resources` subsection at the end of this document.

The video here below illustrates the precipitation measurements acquired by the GPM constellation sensors over a 3-hour period.

..  youtube:: tHXHUc52SAw
   :align: center


.. _gpm_sensors:

GPM Sensors
-------------

The GPM mission relies on passive and active remote sensing measurements to measure the properties of precipitation.
In the following subsections we introduce some theoretical fundamentals of spaceborne radars and passive microwave sensors.


Radars
~~~~~~~~~~~~~~


PMW
~~~~




.. _gpm_data_archive:

GPM Data Archive
-------------------

GPM-API provides tools to easily search files on the `Precipitation Processing System` (PPS)
and the `Goddard Earth Sciences Data and Information Services Center` (GES DISC) data archives
and to download them to your local machine.
However, the PPS and GES DISC data archives can also be explored on your browser.

The following links provide access to the data archives:

- GES DISC TRMM Data: `<https://disc2.gesdisc.eosdis.nasa.gov/data>`_

- GES DISC GPM Data: `<https://gpm1.gesdisc.eosdis.nasa.gov/data>`_

- PPS Research Data: `<https://arthurhouhttps.pps.eosdis.nasa.gov/>`_

- PPS Near-Real-Time Data: `<https://jsimpsonhttps.pps.eosdis.nasa.gov/text/>`_

Please note that the Near-Real-Time (``NRT``) products are available only on the PPS and for a limited time period, typically 5-6 days.
The Research (``RS``) products are instead available on both the PPS and GES DISC with a delay of 2-3 days from NRT.

The Japanese `JAXA G-Portal <https://gportal.jaxa.jp/gpr/?lang=en>`_ facilitates the retrieval of additional data,
including AMSR, AMSR-E L2 products and the GSMaP global precipitation estimates.

Similarly, the Chinese `FengYun Satellite Data Center <https://satellite.nsmc.org.cn/PortalSite/Data/DataView.aspx?currentculture=en-US>`_
provides access to the PMR, MWRI, and MHWHS sensor products.

The GPM-API does not currently support methods for searching, downloading, and opening products from JAXA and FengYun data centers,
but contributions to expand the GPM-API to include these data centers are very welcome !

.. _gpm_product_levels:

GPM Product Levels
-------------------

Satellite data are available in different levels of processing.

- **Level 1A** products provide the unprocessed raw sensor data.

- **Level 1B** products provide the geolocated and radiometrically corrected radar and PMW sensor data.

- **Level 1C** products provides the inter-calibrated PMW brightness temperatures used for generating the L2 PMW products.

- **Level 2A** products contains the geophysical paramaters derived from individual sensors.

- **Level 2B** products contains the geophysical paramaters derived from combined DPR/GMI or PR/TMI sensors.

- **Level 3** gridded products results from the temporal and spatial aggregation of the L2 products.

Currently, the GPM-API provide access to the IMERG products and all L1 and L2 GPM products.
L3 products are currently not available via GPM-API, but can be manually computed using the
Geographic Binning Toolbox provided by the software.

You can retrieve the list of products available through the GPM-API using the ``gpm.available_products()`` function.
For a comprehensive online list of GPM products, refer to `this page <https://gpm.nasa.gov/data/directory>`_
and `this page <https://storm.pps.eosdis.nasa.gov/storm/>`_.

It's important to note that GPM products are available in different versions.
Currently, GPM-API offers access to versions 5, 6, and 7. Version 7 is the latest and is recommended for most applications.

While analyzing a GPM product, it is recommended to consult the corresponding Algorithm Theoretical Basis Document (ATBD) and the
`GPM Products File Specification <https://gpm.nasa.gov/resources/documents/file-specification-gpm-products>`_,
for detailed information on product variables and their attributes.


.. _gpm_products:

GPM Products
-------------------

Here below we provide an introduction to the different types of GPM products.


.. _radar_reflectivity:

Radar Reflectivity
~~~~~~~~~~~~~~~~~~~~~

Radar reflectivity measurements from PR and DPR are available in the L2A products.
Use ``gpm.available_products(category="RADAR", product_levels="2A")`` to list the available products.

.. _pmw_brightness_temperature:

PMW Brightness Temperature
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Brightness temperature measurements from the PMW sensors within the GPM Constellation are available in the L1B and L1C products.
The L1B product provide the raw brightness temperature measurements, while the L1C product provides the inter-calibrated brightness temperatures.
Use ``gpm.available_products(category="PMW", product_levels="1C")`` to list the available products.

.. _radar_precipitation:

Radar Precipitation
~~~~~~~~~~~~~~~~~~~~~




The intensity of instaneous precipitation estimates is capped at 300 mm/hr !**

Precipitation estimates derived from PR and DPR are available in the L2A products.
Use ``gpm.available_products(category="RADAR", product_levels="2A")`` to list the available products.

.. _pmw_precipitation:

PMW Precipitation
~~~~~~~~~~~~~~~~~~~~

Max limit rain PMW ?

Precipitation estimates derived from GPM Constellation sensors are available in the L2A products.
Use ``gpm.available_products(category="PMW", product_levels="2A")`` to list the available products.


.. _corra_precipitation:

CORRA Precipitation
~~~~~~~~~~~~~~~~~~~~~~

The Combined Radar-Radiometer Algorithm (CORRA) use TMI/GMI multichannel radiometric measurements as additional integral constraints on PR/DPR retrieval algorithms.
The combined use of coincident active and passive microwave sensor data offers complementary information about the
macro and microphysical processes of precipitating clouds which can be used to reduce the under constrained nature of the inversion problem.
In simple terms, the combined algorithms use the radiometer signal as a constraint on the attenuation seen by the radar or
to counterbalance the lack of sensitivity of the radar to light precipitation.

CORRA produce a set of geophysical parameters, including hydrometeor profiles, particle size distributions, and surface parameters,
that are physically consistent with both radar reflectivity profiles and brightness temperature radiometer measurements over the radar swath.

**It is important to note that the CORRA instantaneous precipitation rate estimates are currently capped at 80 mm/hr,
while the radar-only products estimates are capped at 300 mm/hr !**

 The TRMM/GPM CORRA product can be accessed with GPM-API using the ``2B-TRMM-CORRA`` and ``2B-GPM-CORRA``  product acronyms.

For the theoretical and mathematical details of the combined radar/radiometer algorithm please
refer to the `CORRA Algorithm Theoretical Basis Document (ATBD) <https://gpm.nasa.gov/resources/documents/gpm-combined-radar-radiometer-precipitation-algorithm-theoretical-basis>`_.


IMERG Precipitation
~~~~~~~~~~~~~~~~~~~~~~

The Integrated Multi-satellite Retrievals for GPM (IMERG) is an advanced algorithm designed
to generate a high resolution precipitation product every 30 minutes, covering the full globe (up to version 6,  a quasi-global area from 60°S to 60°N)
with a spatial resolution of 0.1° x 0.1°.

By leveraging measurements from Geostationary (GEO) IR imagers and the GPM PMW sensors constellation,
IMERG provides a "best-estimate" of 30-minute average precipitation rates.

IMERG generates three distinct precipitation products.
The `IMERG Early` (``IMERG-ER``) product  is available 4 hours after real-time, followed by `IMERG Late` (``IMERG-LR``) after 14 hours.
The final product version, `IMERG Final` (``IMERG-FR``), is released approximately 3.5 months later.

The 4-hour and 14-hour delays for Early and Late products are necessary to gather and process satellite measurements from various space agencies.
The 3.5-month delay for the Final product enables the incorporation of rain gauge data from the Global Precipitation Climatology Centre (GPCC)
and the application of bias correction to refine the satellite-based precipitation estimates.

In the design of IMERG, a significant challenge is the scarcity of direct measurements from the GPM constellation in most grid cells within any 30-minute period.
To fill these spatial and temporal gaps left by PMW satellites, IMERG exploits precipitation estimates derived from GEO IR measurements and cloud-motion vectors.

Cloud-motion vectors enables the backward and forward propagation, or "morphing," of available PMW-derived precipitation measurements across the grid.
While IMERG Early uses only forward propagation in time (extrapolation in time), IMERG Late benefits form from both forward and backward propagation (interpolation in time).

When PMW data are too sparse, IMERG incorporates precipitation estimates derived from GEO IR imagery trough a weighted Kalman filter.

IR-based precipitation estimates are derived from the
`Climate Prediction Center (CPC) Merged 4-km Global IR data product <https://disc.gsfc.nasa.gov/datasets/GPM_MERGIR_1/summary>`_.
This dataset composites infrared (IR) brightness temperature measurements from numerous geostationary sensors over their operational periods,
including European (METEOSAT-5/7/8/9/10/11), Japanese (GMS-5, MTSat-1R/2, Himawari-8/9), and U.S. (GOES-8/9/10/11/12/13/14/15/16/17/18) satellites,
every 30 minutes between 60°N/S.

IR-based precipitation estimates are obtained using the `PERSIANN-CSS algorithm <https://journals.ametsoc.org/view/journals/apme/43/12/jam2173.1.xml>`_ (up to IMERG V6) and
the `PDIR-NOW algorithm <https://journals.ametsoc.org/view/journals/hydr/21/12/jhm-d-20-0177.1.xml>`_ (IMERG V7).
However, these estimates have lower accuracy compared to PMW measurement due to the indirect relationship between infrared cloud top temperature (sensed by the IR imagers)
and surface precipitation.

It's worth noting that the source of cloud-motion vectors has evolved across different IMERG versions:
V5 derived motion vectors from the sequence of GEO IR imagery,
V6 transitioned to using NWP/reanalysis-based total precipitable water vapor fields,
and V7 employs a combination of precipitation, total precipitable liquid water, and water vapor.

For more information on IMERG, including theoretical and algorithmic details, please refer to the Algorithm Theoretical Basis Document (ATBD) of
`IMERG version 6 <https://gpm.nasa.gov/resources/documents/algorithm-information/IMERG-V06-ATBD>`_ and
`IMERG version 7 <https://gpm.nasa.gov/sites/default/files/2023-07/IMERG_V07_ATBD_final_230712.pdf>`_.


Latent Heating
~~~~~~~~~~~~~~~~~

Latent heating refers to the release or absorption of heat energy during phase changes of water within the Earth's atmosphere.

When water vapor condenses into liquid droplets or solid ice crystals, as in the formation of clouds and precipitation,
it releases latent heat into the surrounding atmosphere.
This latent heat, often referred to as "latent heat of condensation" is a primary driver of large- and small-scale atmospheric circulations.
Conversely, when liquid water evaporates into vapor, or solid ice melt to liquid water or sublimates directly into vapor,
heat energy is absorbed from the surrounding environment. This absorption of latent heat is an essential component
of the water cycle and contributes to the redistribution of energy within the atmosphere.

Estimates of the heat released or absorbed at various altitudes are crucial to improve global atmospheric circulation models
and can also be assimilated operationally into numerical weather forecasts to constrain the initial conditions of the models.

Two algorithms, the `Goddard Convective-Stratiform Heating (CSH) <https://gpm.nasa.gov/resources/documents/goddard-convective-stratiform-heating-csh-algorithm>`_ and the
`Japanese Spectral Latent Heating (SLH) <https://www.eorc.jaxa.jp/GPM/doc/algorithm/TRMGPM_SLH_V07A_ATBD.pdf>`_,
are employed for retrieving latent heating esimates from TRMM/GPM measurements.
The SLH algorithm requires only radar data, while CSH make use of the joint PR/TMI and DPR/GMI measurements.

Both algorithms are built upon pre-computed look-up tables derived from simulations from cloud-resolving models (CRMs)
such as the `Goddard Cumulus Ensemble model (GCE) <https://earth.gsfc.nasa.gov/meso/models/gce>`_ and regional-scale models like the
`NASA Unified Weather Research and Forecasting model (NU-WRF) <https://earth.gsfc.nasa.gov/meso/models/nu-wrf>`_ and
the `Japan Meteorological Agency's Local Forecast Model (LFM) <https://www.jma.go.jp/jma/en/Activities/nwp.html>`_.

The TRMM/GPM SLH and CSH products can be accessed with GPM-API using the ``2A-TRMM-SLH``, ``2B-TRMM-CSH``, ``2A-GPM-SLH``, ``2B-GPM-CSH`` product acronyms.

For a recent overview of CSH and SLH products,
please refer to `Tao et al., 2022 <https://www.jstage.jst.go.jp/article/jmsj/100/2/100_2022-015/_html/-char/en>`_ and
`Shige et al., 2009 <https://journals.ametsoc.org/view/journals/clim/22/20/2009jcli2919.1.xml>`_ respectively.


Coincidence datasets
=======================

Here below we detail some of the coincidence satellite datasets available within the TRMM and GPM era.
None of the following datasets is currently supported by GPM-API but we welcome contributions to expand the API to include these datasets.

AMSR-E - CloudSat
~~~~~~~~~~~~~~~~~~~

The `A-Train Constellation <https://atrain.nasa.gov/>`_ is a group of satellites
(including `Aqua <https://aqua.nasa.gov/>`_,
`CloudSat <https://cloudsat.atmos.colostate.edu/>`_,
`CALIPSO <https://www-calipso.larc.nasa.gov/>`_, and
`Aura <https://aura.gsfc.nasa.gov/>`_) that closely followed one another along the same orbital track.

This allowed near-simultaneous observations from a wide variety of instruments that can be
synergistically used to further our understanding of the Earth's atmosphere and surface.

The A-Train Constellation between 2006 and 2011 provided the quasi-simulaneous acquisitions of the
94-GHz (W-band) `CloudSat Profiling Radar (CPR) <https://www.cloudsat.cira.colostate.edu/>`_
and the 12-channel (6.925 - 89 GHz) Advanced Microwave Scanning Radiometer for EOS (AMSR-E).

The `ASMR2-AUX <https://www.cloudsat.cira.colostate.edu/data-products/amsr2-aux>`_ dataset contains a subset
of ancillary AMSR2 surface precipitation and ocean products data collocated with each CPR footprints, while
the `AMSRERR_CPR_002 <https://disc.gsfc.nasa.gov/datasets/AMSRERR_CPR_002/summary>`_ dataset provides only the AMSR-E derived rain rate.

The AMSR-E - CloudSat dataset is particularly useful for developing and validating PMW-based precipitation retrieval algorithms
using CloudSat CPR data as a reference.


TRMM/GPM - CloudSat
~~~~~~~~~~~~~~~~~~~~~

The TRMM/GPM-CloudSat Coincidence dataset collects satellite acquisitions (intersections) occurring
within a small time window between the TRMM/GPM-Core satellites and the 94-GHz (W-band) CloudSat Profiling Radar (CPR).

These coincidences yield combined dual/triple-frequency radar profiles, incluuding W-band from CPR and:

-  the Ku/Ka-band from GPM DPR along with data from the 13-channel (10-183 GHz) GMI radiometer.
-  the Ku-band from TRMM PR along with data from the 9-channel (10-89 GHz) TMI radiometer.

Additionally, the dataset also includes collocated thermal and near-infrared measurements from the
`Moderate Imaging Spectroradiometer (MODIS) <https://aqua.nasa.gov/modis>`_ imager
onboard the Aqua satellite (channels 20 and 27–36).

This dataset serves various scientific purposes, including algorithm evaluation and development, identification of deficiencies,
studies on snow and light rain sensitivity, exploration of cloud processes, radiative transfer simulations, and
examination of land surface effects on radar, radiometer, or combined-sensor precipitation retrieval algorithms.

For detailed information on the TRMM/GPM-CloudSat Coincidence dataset, please
refer to the corresponding `journal article <https://www.mdpi.com/2072-4292/13/12/2264>`_
and the `Algorithm Theoretical Basis Document (ATBD) <https://gpm.nasa.gov/resources/documents/cloudsat-gpm-coincidence-dataset-version-1c>`_.

The coincidence dataset can be download with GPM-API using the ``2B-GPM-CSAT`` and ``2B-TRMM-CSAT`` product acronyms.


GPM-GLM
~~~~~~~~~

The GPM-GLM datasets collocates data from the `Geostationary Lightning Mapper (GLM) <https://www.goes-r.gov/spacesegment/glm.html>`_
instruments onboard of the geostationary satellites GOES-16, GOES-17 and GOES-18 onto the GPM-DPR swath track.
This dataset will expand the lightening record from the `TRMM-LIS <https://gpm.nasa.gov/data/directory/trmm-lis>`_ into the GPM era.

.. warning::

  SOON AVAILABLE


GPM-GEO
~~~~~~~~~~

The GPM-GEO datasets collocates data acquired from the third-generation of geostationary VIS/IR imagers onto the GPM-DPR swath track.
The dataset currently includes L1 (radiances) and L2 (precipitation estimates and cloud properties) derived
from the `Advanced Baseline Imager (ABI) <https://www.goes-r.gov/spacesegment/abi.html>`_ (onboard of GOES-16, GOES-17, GOES-18 satellites)
and from the `Advanced Himawari Imager (AHI) <https://www.data.jma.go.jp/mscweb/en/himawari89/space_segment/spsg_ahi.html>`_ (onboard of Himawari-8 and Himawari-9 satellites).

.. warning::

  SOON AVAILABLE


Feature database
=======================

Here below we provide details about several "feature" database derived from the TRMM and GPM sensors.
While these datasets are not currently supported by the GPM-API, we welcome contributions to expand the API to include them.

TAMU-CC Precipitation Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pioneered by `Nesbitt et al. in 2000 <https://journals.ametsoc.org/view/journals/apme/47/10/2008jamc1890.1.xml>`_ and
further formalized by `Liu et al. in 2008 <https://journals.ametsoc.org/view/journals/clim/13/23/1520-0442_2000_013_4087_acopfi_2.0.co_2.xml>`_,
this database includes Precipitation Features(PFs) derived from TRMM (PR, TMI, VIRS, and LIS) and GPM-CO (DPR, GMI) sensors.

The `website of Chuntao Liu <http://atmos.tamucc.edu/trmm/>`_ provides access to the data and additional
useful resources to gain insight into the `climatology of precipitation features <http://atmos.tamucc.edu/trmm/p_pfs.html>`_.


UW Storm Classification
~~~~~~~~~~~~~~~~~~~~~~~~~

The UW storm classification system (`Houze et al., 2015 <https://agupubs.onlinelibrary.wiley.com/doi/10.1002/2015RG000488>`)
is based on the three-dimensional storm structure measured by the TRMM and GPM radars.
It categorizes storms into five types: Isolated Shallow Echoes (ISE), Deep Convective Cores (DCC), Wide Convective Cores (WCC),
Deep-Wide Convective Cores (DWC), and Broad Stratiform Rain areas (BSR).

Data are available for `TRMM here <http://trmm.atmos.washington.edu/>`_  and `GPM here <http://gpm.atmos.washington.edu/>`_


.. _satellite_precipitation_measurements:

Real-Time Instantanuous Measurements
-------------------------------------

If you're interested in exploring near-surface precipitation data in almost real-time from the
GPM DPR or GMI sensors, the `JAXA GPM Real-Time Monitor <https://sharaku.eorc.jaxa.jp/trmm/RT3/index.html>`_ is a valuable tool.
This platform also allows for the visualization of data from the TRMM PR and TMI going back to 1998.

This tool makes it easy to identify when and where the GPM satellites are detecting precipitation.

If you spot a precipitating system on the monitor that interests you, activating the  ``Observation Time`` toggle on the
lower left will enable you to obtain the sensor's acquisition time with minute-level accuracy.

By copying such acquisition time, you can easily download, analyze and visualize the corresponding data using the GPM API.

The GIF and code snippet here below showcases the step-by-step process for identifying an interesting precipitation event,
copying its acquisition time, and leveraging the GPM API for data visualization and analysis.


.. code-block:: python

    import gpm

    product = "2A-DPR"
    product_type = "NRT"  # if ~48 h from real-time data, otherwise "RS" (Research) ...
    version = 7

    start_time = datetime.datetime(2020, 7, 22, 1, 10, 11)
    end_time = datetime.datetime(2020, 7, 22, 2, 30, 5)

    # Download data over specific time periods
    gpm.download(
        product=product,
        product_type=product_type,
        version=version,
        start_time=start_time,
        end_time=end_time,
    )
    # Open the dataset
    ds = gpm.open_dataset(
        product=product,
        product_type=product_type,
        version=version,
        start_time=start_time,
        end_time=end_time,
    )

    # Plot a specific variable of the dataset
    ds["precipRateNearSurface"].gpm.plot_map()


If you're interested in measurements from other satellites, the `JAXA Global Rainfall Watch <https://sharaku.eorc.jaxa.jp/GSMaP/index.htm>`_
allows you to visualize the GPM PMW constellation swath coverage over a 1-hour period.
This is achieved by activating the ``Time and Satellite`` toggle located in the top right corner of the interface.

.. _tropical_cyclones_measurements:

Tropical Cyclones Measurements
-------------------------------

The JAXA-EORC Tropical Cyclones `Real Time Monitoring <https://sharaku.eorc.jaxa.jp/cgi-bin/typhoon_rt/main.cgi?lang=en>`_
and `Database <https://sharaku.eorc.jaxa.jp/TYP_DB/index.html>`_ websites provides quicklooks of
the latest and past tropical cyclones satellite acquisitions of DPR, GMI and AMSR2 sensors.


TC-PRIMED
~~~~~~~~~~~

If you are interested in tropical cyclones studies using PMW data, please also have a look at the
`TC-PRIMED dataset <https://rammb-data.cira.colostate.edu/tcprimed/>`_.
TC-PRIMED contains over 197'000 PMW overpasses of 2'300 global tropical cyclones from 1998 to 2021.


TC-PF
~~~~~~~~~~~

The `Tropical Cyclone related Precipitation Feature` (TC-PF) database is a subset of the `TAMU-CC Precipitation Features` database.
Please refer to `Jiang et al., 2011 <https://journals.ametsoc.org/view/journals/apme/50/6/2011jamc2662.1.xml>`_ for more information on the TC-PF database.
Data are available at `this link <https://ghrc.nsstc.nasa.gov/pub/tcpf/tcpf-L1/data/>`_.


.. _global_precipitation_products:

Global Precipitation Products
------------------------------

The state-of-the-art global precipitation satellite-based products available with a temporal resolution
of 30 minutes and 0.1° and 0.1° spatial resolutions are GSMAP and IMERG.
These products are based on the merging of multiple satellite sensors, including PMW and VIS/IR sensors.

Based on the type of applications your are interested in, you may also want to consider other products, such as PERSIANN, MSWEP, and ERA5.
Here below we provide a table summarizing some high-quality global precipitation products.

+------------+---------------------------------------------------------------------------------+---------------------+--------------------+--------------------------------------------------------------------------------------------------+
| Acronym    | Full Name                                                                       | Temporal Resolution | Spatial Resolution | Data Source                                                                                      |
+============+=================================================================================+=====================+====================+==================================================================================================+
| IMERG      | Integrated Multi-satellitE Retrievals for GPM                                   | 30 minutes          | 0.1°               | `NASA <https://gpm.nasa.gov/data/imerg>`_                                                        |
+------------+---------------------------------------------------------------------------------+---------------------+--------------------+--------------------------------------------------------------------------------------------------+
| GSMaP      | Global Satellite Mapping of Precipitation                                       | 30 minutes          | 0.1°               | `JAXA <https://sharaku.eorc.jaxa.jp/GSMaP/guide.html>`_                                          |
+------------+---------------------------------------------------------------------------------+---------------------+--------------------+--------------------------------------------------------------------------------------------------+
| PERSIANN   | Precipitation Estimation from Remotely Sensed Information using Artificial NNs  | 1 hour              | 0.04°              | `CHRS <https://chrsdata.eng.uci.edu/>`_                                                          |
+------------+---------------------------------------------------------------------------------+---------------------+--------------------+--------------------------------------------------------------------------------------------------+
| MSWEP      | Multi-Source Weighted-Ensemble Precipitation                                    | 3 hour              | 0.1°               | `GloH2O <https://www.gloh2o.org/mswep/>`_                                                        |
+------------+---------------------------------------------------------------------------------+---------------------+--------------------+--------------------------------------------------------------------------------------------------+
| ERA5       | ERA5 Reanalysis                                                                 | 1 hour              | 0.1°               | `ECMWF <https://cds.climate.copernicus.eu/cdsapp#!/dataset/10.24381/cds.e2161bac?tab=overview>`_ |
+------------+---------------------------------------------------------------------------------+---------------------+--------------------+--------------------------------------------------------------------------------------------------+

GPM-API currently provides only access to the version 6 and 7 of the IMERG products, and the variable ``IRprecipitation`` in such IMERG products results from the PERSIANN-CCS and PDIR-Now algorithms respectively.

We welcome contributions that enable GPM-API to access other precipitation products !

GSMaP V6 is also available On Google Earth Engine: https://developers.google.com/earth-engine/datasets/catalog/JAXA_GPM_L3_GSMaP_v6_operational
IMERG V6 is also available on Google Earth Engine: https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_V06

GSMaP can be visualized on the `JAXA Global Rainfall Watch <https://sharaku.eorc.jaxa.jp/GSMaP/index.htm>`_,
while IMERG on the `GPM IMERG Global Viewer <https://gpm.nasa.gov/data/visualization/global-viewer>`_ or the
`EOSDIS WorldView Portal <https://worldview.earthdata.nasa.gov/?v=-235.13866988428558,-76.35016978404038,104.5800850894752,96.99821113230026&l=Reference_Labels_15m(hidden),Reference_Features_15m(hidden),Coastlines_15m,IMERG_Precipitation_Rate,VIIRS_NOAA20_CorrectedReflectance_TrueColor(hidden),VIIRS_SNPP_CorrectedReflectance_TrueColor(hidden),MODIS_Aqua_CorrectedReflectance_TrueColor(hidden),MODIS_Terra_CorrectedReflectance_TrueColor&lg=true&t=2024-02-08-T03%3A43%3A10Z>`_.

.. _useful_resources:

Useful Resources
------------------

For those seeking detailed information and resources related to the GPM Mission
and associated satellite measurements, the following table organizes key links to FAQs, training materials,
specific mission pages and ATBDs.

This compilation provides a comprehensive starting point for researchers, students,
and enthusiasts to explore educational resources and technical details.


.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - Resource Type
     - Description
     - URLs
   * - Training
     - Additional information and training resources
     - | `NASA Materials <https://gpm.nasa.gov/data/training>`_
       | `JAXA Materials <https://www.eorc.jaxa.jp/GPM/en/materials.html>`_
       | `REMSS Materials <https://www.remss.com/>`_
   * - GPM News
     - News related to the GPM mission
     - | `JAXA GPM News <https://www.eorc.jaxa.jp/GPM/en/index.html>`_
       | `NASA GPM News <https://gpm.nasa.gov/data/news>`_
   * - GPM FAQ
     - Frequently Asked Questions about GPM data
     - `GPM FAQ <https://gpm.nasa.gov/data/faq>`_
   * - GPM Mission
     - Global Precipitation Measurement Mission
     - | `NASA GPM <https://gpm.nasa.gov/missions>`_
       | `JAXA GPM <https://www.eorc.jaxa.jp/GPM/en/index.html>`_
       | `eoPortal GPM <https://www.eoportal.org/satellite-missions/gpm>`_
   * - TRMM Mission
     - Tropical Rainfall Measuring Mission
     - | `NASA TRMM <https://trmm.gsfc.nasa.gov/>`_
       | `JAXA TRMM <https://www.eorc.jaxa.jp/TRMM/index_e.htm>`_
       | `eoPortal TRMM <https://www.eoportal.org/satellite-missions/trmm>`_
   * - ATBDs
     - Algorithm Theoretical Basis Documents
     - | `GPM Documents <https://gpm.nasa.gov/resources/documents>`_
       | `JAXA Documents <https://www.eorc.jaxa.jp/GPM/en/archives.html>`_


.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - GPM PMW sensors
     - Full Name
     - URLs
   * - AMSR-E
     - Advanced Microwave Scanning Radiometer-EOS
     - | `JAXA AMSR-E <https://sharaku.eorc.jaxa.jp/AMSR/index.html>`_
       | `eoPortal AMSR-E <https://www.eoportal.org/satellite-missions/aqua#amsr-e-advanced-microwave-scanning-radiometer-eos>`_
   * - AMSR2
     - Advanced Microwave Scanning Radiometer 2
     - | `JAXA AMSR2 <https://www.eorc.jaxa.jp/AMSR/index_en.html>`_;
       | `eoPortal AMSR2 <https://www.eoportal.org/satellite-missions/gcom#amsr2-advanced-microwave-scanning-radiometer-2>`_
   * - AMSU-B
     - Advanced Microwave Sounding Unit-B
     - `eoPortal AMSU-B <https://www.eoportal.org/satellite-missions/noaa-poes-series-5th-generation#amsu-b-advanced-microwave-sounding-unit---b>`_
   * - ATMS
     - Advanced Technology Microwave Sounder
     - | `NOAA ATMS Website <https://www.nesdis.noaa.gov/our-satellites/currently-flying/joint-polar-satellite-system/advanced-technology-microwave-sounder-atms>`_
       | `eoPortal ATMS <https://www.eoportal.org/satellite-missions/atms>`_
   * - MHS
     - Microwave Humidity Sounder
     - `eoPortal MHS Summary <https://www.eoportal.org/satellite-missions/metop#mhs-microwave-humidity-sounder>`_
   * - SAPHIR
     - Sondeur Atmospherique du Profil d'Humidite Intertropicale par Radiometrie
     - | `Megha-Tropiques <https://meghatropiques.ipsl.fr/>`_
       | `eoPortal SAPHIR <https://www.eoportal.org/satellite-missions/megha-tropiques#saphir-sondeur-atmospherique-du-profil-dhumidite-intertropicale-par-radiometries>`_
   * - SSMIS
     - Special Sensor Microwave - Imager/Sounder
     - `eoPortal SSMIS <https://www.eoportal.org/satellite-missions/dmsp-block-5d#ssmis-special-sensor-microwave-imager-sounder>`_


.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - Other PMW sensors
     - Full Name
     - URLs
   * - GEMS
     - Global Environmental Monitoring System
     - | `WeatherStream GEMS <https://weatherstream.com/gems/>`_
       | `eoPortal IOD-1 GEMS <https://www.eoportal.org/satellite-missions/iod-1-gems#references>`_
   * - MTVZA
     - Microwave Imaging/Sounding Radiometer
     - | `eoPortal Meteor-M MTVZA <https://www.eoportal.org/satellite-missions/meteor-m-1#mtvza-gy-microwave-imagingsounding-radiometer>`_
       | `eoPortal Meteor-3M MTVZA <https://www.eoportal.org/satellite-missions/meteor-3m-1#mtvza-microwave-imagingsounding-radiometer>`_
   * - MWHS
     - Microwave Humidity Sounder
     - | `NSMC MWHS <https://fy4.nsmc.org.cn/nsmc/en/instrument/MWHS.html>`_
       | `eoPortal FY-3 MWHS <https://www.eoportal.org/satellite-missions/fy-3#mwhs-microwave-humidity-sounder>`_
   * - MWRI
     - Microwave Radiometer Imager
     - | `NSMC MWRI <https://www.nsmc.org.cn/nsmc/en/instrument/MWRI.html>`_
       | `NSMC/GSICS Monitoring <http://gsics.nsmc.org.cn/portal/en/monitoring/MWRI.html>`_
       | `eoPortal FY-3 MWRI <https://www.eoportal.org/satellite-missions/fy-3#mwri-microwave-radiometer-imager>`_
       | `eoPortal HY-2A MWRI <https://www.eoportal.org/satellite-missions/hy-2a#mwri-microwave-radiometer-imager>`_
   * - TEMPEST-D
     - Temporal Experiment for Storms and Tropical Systems Demonstration
     - | `CSU TEMPEST <https://tempest.colostate.edu/>`_
       | `eoPortal TEMPEST-D <https://www.eoportal.org/satellite-missions/tempest-d#launch>`_
   * - Tomorrow.io Sounder
     - Tomorrow.io Sounder
     - `Tomorrow.io Sounder <https://www.tomorrow.io/space/sounder/>`_
   * - TROPICS
     - Time-Resolved Observations of Precipitation structure and storm Intensity with a Constellation of Smallsats
     - | `MIT TROPICS <https://tropics.ll.mit.edu/CMS/tropics/>`_
       | `NASA TROPICS <https://weather.ndc.nasa.gov/tropics/>`_
       | `eoPortal TROPICS <https://www.eoportal.org/satellite-missions/tropics>`_
   * - WindSat
     - WindSat Polarimetric Microwave Radiometer
     - `eoPortal WindSat <https://www.eoportal.org/satellite-missions/coriolis#mission-status>`_

.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - Other radar sensors
     - Full Name
     - URLs

   * - PMR
     - Feng Yun Precipitation Measurement Radar
     - | `Zhang et al., 2023 <https://spj.science.org/doi/10.34133/remotesensing.0097>`_
       | `NSMC PMR <https://www.nsmc.org.cn/nsmc/en/instrument/PMR.html>`_
       | `NSMC Monitoring <http://gsics.nsmc.org.cn/portal/en/monitoring/PMR.html>`_
   * - RainCube
     - Radar in a CubeSat
     - | `JPL RainCube <https://www.jpl.nasa.gov/missions/radar-in-a-cubesat-raincube>`_
       | `eoPortal RainCube <https://www.eoportal.org/satellite-missions/raincube#development-status>`_
   * - Tomorrow R1 and R2
     - Tomorrow.io's Radar
     - | `Tomorrow.io Radar <https://www.tomorrow.io/space/radar-satellites>`_
       | `eoPortal Tomorrow R1 and R2 <https://www.eoportal.org/satellite-missions/tomorrow-r1-r2#references>`_
