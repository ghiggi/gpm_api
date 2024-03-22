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
was initiated following the success of the Tropical Rainfall Measuring Mission (TRMM) satellite launched in November 1997.

The GPM mission aims to acquire accurate and frequent global observations of Earth's precipitation.
The mission relies on the GPM Core Observatory satellite and a constellation of satellites equipped
with microwave radiometers which forms the so-called GPM Constellation.

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

TRMM mission
---------------

The TRMM satellite, launched in November 1997 as a collaborative effort between the
United States' National Aeronautics and Space Administration (NASA) and the
Japan Aerospace Exploration Agency (JAXA),
marked a significant milestone for the satellite remote sensing observation of tropical precipitations between 37°N/S.
Its primary objective was to accurately measure rainfall associated with tropical convective activity, which play a crucial role in the global dynamics of the atmospheric circulation.

Equipped with pioneering technology including the first spaceborne Precipitation Radar (PR), the TRMM Microwave Imager (TMI), the Visible and Infrared Scanner (VIRS) imager,
and the Lightning Imaging Sensor (LIS), TRMM aimed to revolutionize rainfall observation.

The combined use of PR and TMI significantly enhanced rainfall estimation accuracy over the tropics and subtropics.
Moreover, PR provided unprecedented insights into the three-dimensional structure of cyclones over the ocean,
as well as rainfall characteristics of the Madden-Julian Oscillation and other climate phenomena such as El Niño and La Niña.

The success of the TRMM mission underscored the potential of satellite remote sensing in advancing our understanding of Earth's water cycle
and improving weather forecasting capabilities.


.. _gpm_core_satellite:

GPM-Core Observatory
---------------------

The GPM Core Observatory satellite, a joint collaboration between NASA and JAXA, was launched on February 28, 2014,
building upon the legacy of the TRMM to extend precipitation measurement capabilities from the tropics to higher latitudes (68°N/S).

Equipped with advanced instruments such as the Dual-frequency Precipitation Radar (DPR) and the GPM Microwave Imager (GMI),
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


.. _gpm_gmi:

GPM GMI
~~~~~~~~

TODO



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


.. _gpm_products:

GPM Products
-------------------

TODO: provide intro on global precipitation products
Microwave, VIS/IR, multi-satellite merging

CloudSat/AMSR-E collocated (Aqua satellite) (2002-2011)
Coincidence dataset TRMM-CloudSat GPM-CloudSat

Radar products
~~~~~~~~~~~~~~

PMW products
~~~~~~~~~~~~~~

Latent Heating products
~~~~~~~~~~~~~~~~~~~~~~~~~

IMERG products
~~~~~~~~~~~~~~





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


.. _satellite_precipitation_measurements:

Satellite Precipitation Measurements
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

If you are interested in tropical cyclones studies using PMW data, please also have a look at the
`TC-PRIMED dataset <https://rammb-data.cira.colostate.edu/tcprimed/>`_.
TC PRIMED contains over 197'000 PMW overpasses of 2'300 global tropical cyclones from 1998 to 2021.


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
