=================
Introduction
=================

Skip to what's is interest for you
Bullet point list with reference


GPM
-------------------

.. warning::

  SOON AVAILABLE

Radar


Passive Microwave Sensors





Global Precipitation Products
------------------------------

TODO: provide intro on global precipitation products
Microwave, VIS/IR, multi-satellite merging

The state-of-the-art global precipitation satellite-based products available with a temporal resolution of 30 minutes and 0.1° and 0.1° spatial resolutions are
GSMAP, IMERG and
Acronym, Full Name, Temporal Resolution, Spatial Resolution, Data Source

`Integrated Multi-satellitE Retrievals for GPM (IMERG) <https://gpm.nasa.gov/data/imerg>`_
`Global Satellite Mapping of Precipitation (GSMaP) <https://sharaku.eorc.jaxa.jp/GSMaP/guide.html>`_
`Precipitation Estimation from Remotely Sensed Information using Artificial Neural Networks (PERSIANN) <https://chrsdata.eng.uci.edu/>`_'
`Multi-Source Weighted-Ensemble Precipitation (MSWEP) <https://www.gloh2o.org/mswep/>`_
`ERA5 Reanalysis <https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=overview>`_

GPM-API currently provides access only to IMERG V7, which also include a version of CMORPH and PERSIANN products.
We welcome contributions that enable GPM-API to access and open other precipitation products.!

GSMaP V6 is also available On Google Earth Engine: https://developers.google.com/earth-engine/datasets/catalog/JAXA_GPM_L3_GSMaP_v6_operational
IMERG V6 is also available on Google Earth Engine: https://developers.google.com/earth-engine/datasets/catalog/NASA_GPM_L3_IMERG_V06

GSMaP can be visualized on the `JAXA Global Rainfall Watch <https://sharaku.eorc.jaxa.jp/GSMaP/index.htm>`_, while IMERG on
the `GPM IMERG Global Viewer <https://gpm.nasa.gov/data/visualization/global-viewer>`_ or the
`EOSDIS WorldView Portal <https://worldview.earthdata.nasa.gov/?v=-235.13866988428558,-76.35016978404038,104.5800850894752,96.99821113230026&l=Reference_Labels_15m(hidden),Reference_Features_15m(hidden),Coastlines_15m,IMERG_Precipitation_Rate,VIIRS_NOAA20_CorrectedReflectance_TrueColor(hidden),VIIRS_SNPP_CorrectedReflectance_TrueColor(hidden),MODIS_Aqua_CorrectedReflectance_TrueColor(hidden),MODIS_Terra_CorrectedReflectance_TrueColor&lg=true&t=2024-02-08-T03%3A43%3A10Z>`_.


GPM Products
-------------------

.. warning::

  SOON AVAILABLE

https://gpm.nasa.gov/data/directory
https://storm.pps.eosdis.nasa.gov/storm/
L3 data are not yet available with GPM-API (except for IMERG products)
They can manually be computed using the Geographic Binning Tool provided by the software.

GPM Products File Specification
https://gpm.nasa.gov/resources/documents/file-specification-gpm-products



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


The Japanese `JAXA G-Portal <https://gportal.jaxa.jp/gpr/?lang=en>`_ facilitates the retrieval of additional data,
including AMSR, AMSR-E L2 products and the GSMaP global precipitation estimates.

Similarly, the Chinese `FengYun Satellite Data Center <https://satellite.nsmc.org.cn/PortalSite/Data/DataView.aspx?currentculture=en-US>`_
provides access to the PMR, MWRI, and MHWHS sensor products.

However, the GPM-API does not currently support methods for searching, downloading, and opening products from these data centers.
Contributions to expand the GPM-API to include these data centers are very welcome !


Satellite Measurements in Near-Real-Time
----------------------------------------------

If you're interested in exploring near-surface precipitation data in almost real-time from the
GPM's Dual-frequency Precipitation Radar (DPR) and/or GPM Microwave Imager (GMI),
the `JAXA GPM Real-Time Monitor <https://sharaku.eorc.jaxa.jp/trmm/RT3/index.html>`_ is a valuable tool.
This platform also allows for the visualization of data from the TRMM's Microwave Imager (TMI) and Precipitation Radar (PR) going back to 1998.

This tool makes it easy to identify when and where the GPM satellites are detecting precipitation.

If you spot a precipitating system on the monitor that interests you, activating the "Observation Time" toggle on the
lower left will enable you to obtain the sensor's acquisition time with minute-level accuracy.

By copying such acquisition time, you can easily download, analyze and visualize the corresponding data using the GPM API.

The GIF and code snippet here below showcases the step-by-step process for identifying an interesting precipitation event,
copying its acquisition time, and leveraging the GPM API for data visualization and analysis.


.. code-block:: python

    import gpm_api

    product = "2A-DPR"
    product_type = "NRT"  # if ~48 h from real-time data, otherwise "RS" (Research) ...
    version = 7
    storage = "pps"  # or "ges_disc"

    start_time = datetime.datetime(2020, 7, 22, 1, 10, 11)
    end_time = datetime.datetime(2020, 7, 22, 2, 30, 5)

    # Download data over specific time periods
    gpm_api.download(
        product=product,
        product_type=product_type,
        version=version,
        start_time=start_time,
        end_time=end_time,
        storage=storage,
    )
    ds = gpm_api.open_dataset(
        product=product,
        product_type=product_type,
        version=version,
        start_time=start_time,
        end_time=end_time,
    )

    # Plot a specific variable of the dataset
    ds["Tc"].gpm_api.plot_map()


If you're interested in measurements from other satellites, the `JAXA Global Rainfall Watch <https://sharaku.eorc.jaxa.jp/GSMaP/index.htm>`_
allows you to visualize the Passive Microwave (PMW) acquisitions of the entire GPM constellation over a 1-hour period.
This is achieved by activating the 'Time and Satellite' toggle located in the top right corner of the interface.


Tropical Cyclone Measurements in Near-Real-Time
------------------------------------------------

The JAXA-EORC Tropical Cyclones `Real Time Monitoring <https://sharaku.eorc.jaxa.jp/cgi-bin/typhoon_rt/main.cgi?lang=en>`_
and `Database <https://sharaku.eorc.jaxa.jp/TYP_DB/index.html>`_ websites provides quicklooks of
the latest and past tropical cyclones satellite acquisitions of DPR, GMI and AMSR2 sensors.

If you are interested in tropical cyclones studies using PMW data, please also have a look at the
`TC-PRIMED dataset <https://rammb-data.cira.colostate.edu/tcprimed/>`_.
TC PRIMED contains over 197'000 PMW overpasses of 2'300 global tropical cyclones from 1998 to 2021.



Useful Resources
------------------

For those seeking detailed information and resources related to the Global Precipitation Measurement (GPM) Mission and associated satellite measurements,
the following table organizes key links to FAQs, training materials, Algorithm Theoretical Basis Documents (ATBDs), and specific mission pages.
This compilation provides a comprehensive starting point for researchers, students, and enthusiasts to explore data, technical details, and educational resources.

.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - Resource Type
     - Description
     - URLs
   * - Training
     - Additional information and training resources
     - `GPM Materials <https://gpm.nasa.gov/data/training>`_;
       `JAXA Materials <https://www.eorc.jaxa.jp/GPM/en/materials.html>`_;
       `REMSS Materials <https://www.remss.com/>`_
   * - GPM News
     - News related to the GPM mission
     - `JAXA GPM News <https://www.eorc.jaxa.jp/GPM/en/index.html>`_
       `NASA GPM News <https://gpm.nasa.gov/data/news>`_
   * - GPM FAQ
     - Frequently Asked Questions about GPM data
     - `GPM FAQ <https://gpm.nasa.gov/data/faq>`_
   * - ATBD
     - Algorithm Theoretical Basis Documents
     - `GPM Documents <https://gpm.nasa.gov/resources/documents>`_;
       `JAXA Documents <https://www.eorc.jaxa.jp/GPM/en/archives.html>`_
   * - GPM Mission
     - Global Precipitation Measurement Mission
     - `NASA GPM Website <https://gpm.nasa.gov/missions>`_;
       `JAXA GPM Website <https://www.eorc.jaxa.jp/GPM/en/index.html>`_;
       `eoPortal GPM Summary <https://www.eoportal.org/satellite-missions/gpm>`_
   * - TRMM Mission
     - Tropical Rainfall Measuring Mission
     - `NASA TRMM Website <https://trmm.gsfc.nasa.gov/>`_;
       `JAXA TRMM Website <https://www.eorc.jaxa.jp/TRMM/index_e.htm>`_;
       `eoPortal TRMM Summary <https://www.eoportal.org/satellite-missions/trmm>`_


.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - GPM Constellation PMW sensors
     - Full Name
     - URLs
   * - AMSR-E
     - Advanced Microwave Scanning Radiometer-EOS
     - `JAXA AMSR-E Website <https://sharaku.eorc.jaxa.jp/AMSR/index.html>`_;
       `eoPortal AMSR-E Summary <https://www.eoportal.org/satellite-missions/aqua#amsr-e-advanced-microwave-scanning-radiometer-eos>`_
   * - AMSR2
     - Advanced Microwave Scanning Radiometer 2
     - `JAXA AMSR2 Website <https://www.eorc.jaxa.jp/AMSR/index_en.html>`_;
       `eoPortal AMSR2 Summary <https://www.eoportal.org/satellite-missions/gcom#amsr2-advanced-microwave-scanning-radiometer-2>`_
   * - AMSU-B
     - Advanced Microwave Sounding Unit-B
     - `eoPortal AMSU-B Summary <https://www.eoportal.org/satellite-missions/noaa-poes-series-5th-generation#amsu-b-advanced-microwave-sounding-unit---b>`_
   * - ATMS
     - Advanced Technology Microwave Sounder
     - `NOAA ATMS Website <https://www.nesdis.noaa.gov/our-satellites/currently-flying/joint-polar-satellite-system/advanced-technology-microwave-sounder-atms>`_;
       `eoPortal ATMS Summary <https://www.eoportal.org/satellite-missions/atms>`_
   * - MHS
     - Microwave Humidity Sounder
     - `eoPortal MHS Summary <https://www.eoportal.org/satellite-missions/metop#mhs-microwave-humidity-sounder>`_
   * - SAPHIR
     - Sondeur Atmospherique du Profil d'Humidite Intertropicale par Radiometrie
     - `Megha-Tropiques Website <https://meghatropiques.ipsl.fr/>`_;
       `eoPortal SAPHIR Summary <https://www.eoportal.org/satellite-missions/megha-tropiques#saphir-sondeur-atmospherique-du-profil-dhumidite-intertropicale-par-radiometries>`_
   * - SSMIS
     - Special Sensor Microwave - Imager/Sounder
     - `eoPortal SSMIS Summary <https://www.eoportal.org/satellite-missions/dmsp-block-5d#ssmis-special-sensor-microwave-imager-sounder>`_


.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - Other PMW sensors
     - Full Name
     - URLs
   * - GEMS
     - Global Environmental Monitoring System
     - `WeatherStream GEMS Website <https://weatherstream.com/gems/>`_;
       `eoPortal IOD-1 GEMS Summary <https://www.eoportal.org/satellite-missions/iod-1-gems#references>`_
   * - MTVZA
     - Microwave Imaging/Sounding Radiometer
     - `eoPortal Meteor-M MTVZA Summary <https://www.eoportal.org/satellite-missions/meteor-m-1#mtvza-gy-microwave-imagingsounding-radiometer>`_;
       `eoPortal Meteor-3M MTVZA Summary <https://www.eoportal.org/satellite-missions/meteor-3m-1#mtvza-microwave-imagingsounding-radiometer>`_
   * - MWHS
     - Microwave Humidity Sounder
     - `NSMC MWHS Website <https://fy4.nsmc.org.cn/nsmc/en/instrument/MWHS.html>`_;
       `eoPortal FY-3 MWHS Summary <https://www.eoportal.org/satellite-missions/fy-3#mwhs-microwave-humidity-sounder>`_
   * - MWRI
     - Microwave Radiometer Imager
     - `NSMC MWRI Website <https://www.nsmc.org.cn/nsmc/en/instrument/MWRI.html>`_;
       `NSMC/GSICS Monitoring Website <http://gsics.nsmc.org.cn/portal/en/monitoring/MWRI.html>`_;
       `eoPortal FY-3 MWRI Summary <https://www.eoportal.org/satellite-missions/fy-3#mwri-microwave-radiometer-imager>`_;
       `eoPortal HY-2A MWRI Summary <https://www.eoportal.org/satellite-missions/hy-2a#mwri-microwave-radiometer-imager>`_
   * - TEMPEST-D
     - Temporal Experiment for Storms and Tropical Systems Demonstration
     - `Colorado State University TEMPEST Website <https://tempest.colostate.edu/>`_;
       `eoPortal TEMPEST-D Summary <https://www.eoportal.org/satellite-missions/tempest-d#launch>`_
   * - TROPICS
     - Time-Resolved Observations of Precipitation structure and storm Intensity with a Constellation of Smallsats
     - `MIT TROPICS Website <https://tropics.ll.mit.edu/CMS/tropics/>`_;
       `NASA TROPICS Website <https://weather.ndc.nasa.gov/tropics/>`_;
       `eoPortal TROPICS Summary <https://www.eoportal.org/satellite-missions/tropics>`_
   * - WindSat
     - WindSat Polarimetric Microwave Radiometer
     - `eoPortal WindSat Summary <https://www.eoportal.org/satellite-missions/coriolis#mission-status>`_

.. list-table::
   :widths: 25 50 25
   :header-rows: 1

   * - Other radar sensors
     - Full Name
     - URLs

   * - PMR
     - Feng Yun Precipitation Measurement Radar
     - `Zhang et al., 2023 <https://spj.science.org/doi/10.34133/remotesensing.0097>`_;
       `NSMC Website <https://www.nsmc.org.cn/nsmc/en/instrument/PMR.html>`_;
       `NSMC Monitoring <http://gsics.nsmc.org.cn/portal/en/monitoring/PMR.html>`_;
   * - RainCube
     - Radar in a CubeSat
     - `JPL RainCube <https://www.jpl.nasa.gov/missions/radar-in-a-cubesat-raincube>`_;
       `eoPortal RainCube Summary <https://www.eoportal.org/satellite-missions/raincube#development-status>`_
   * - Tomorrow R1 and R2
     - Tomorrow.io's Radar
     - `Tomorrow.io Website: <https://www.tomorrow.io/space/radar-satellites>`_;
       `eoPortal Tomorrow R1 and R2 <https://www.eoportal.org/satellite-missions/tomorrow-r1-r2#references>`_
