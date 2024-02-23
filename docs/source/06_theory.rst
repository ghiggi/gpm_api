=========================
Remote Sensing Theory
=========================

OR Applications

Radar

The unique function of precipitation radars is to provide the three-dimensional structure of rainfall, obtaining high quality rainfall estimates over ocean and land. Radar measurements are typically less sensitive to the surface and provide a nearly direct relationship between radar reflectivities and the physical characteristics of the rain and snow in a cloud. Because of the complexities of operating radar in space, limited channels (frequencies) are designed for the instruments.

The TRMM satellite had a single frequency radar at the Ku-band particularly sensitive to moderate rain rates. With a single frequency, the TRMM radar was able to retrieve one parameter of the rain drop particle size distribution (PSD); either the median drop size or the number of drops at each range gate bin in the vertically sampled profile (e.g., each 500 meters vertically in the cloud). The GPM Dual-frequency Precipitation Radar has two frequencies, Ku-band like TRMM, and Ka-band that is sensitive to lighter rain and falling snow; so both drop size and distribution of drops can be retrieved. The algorithm requires assumptions and corrections to retrieve precipitation structure, most notably to correct for attenuation. Attenuation refers to the weakening of a radar signal as it moves further away from the emitter and is scattered, reflected, and absorbed with by precipitation and other atmospheric particles.

https://gpm.nasa.gov/resources/documents/gpm-dpr-level-2-algorithm-theoretical-basis-document-atbd

PMW

Precipitation radiometers provide additional degrees of freedom for interpreting rain and snow in clouds through the use of multiple passive frequencies (9 for TRMM and 13 for GPM).
Brightness temperatures at each frequency are a measure of everything in their field of view.
These frequencies transition from being sensitive to liquid rain drops at the low (10 GHz) end to being sensitive to the snow and
ice particles at the middle (91 GHz) and high (183 GHz) end. So, simplifying, when there is liquid rain in the cloud column,
the low frequency channels will respond; when there is snow the high frequency channels will respond;
when there is clear air the brightness temperatures respond to the surface emission.
In lighter rain, the surface emission may contaminate the brightness temperatures such that additional information
is needed to constrain precipitation retrievals.
Specifically, over land only the higher frequencies are operationally useful for estimating precipitation.

Retrievals from passive sensor measurements typically rely on a priori information, such as that used in Bayesian databases to reduce assumptions. This a priori information of multi-frequency brightness temperatures linked to precipitation profiles can be generated from models of clouds and calculations of brightness temperatures (as was done for TRMM) or from combined active-passive retrievals (as is done for GPM). The retrievals typically perform a best match between the multi-frequency observations and the Bayesian database TBs.
https://gpm.nasa.gov/resources/documents/gpm-gprof-algorithm-theoretical-basis-document-atbd

CMB
The combined use of coincident active and passive microwave sensor data provides complementary information about the macro and microphysical processes of precipitating clouds which can be used to reduce uncertainties in combined radar/radiometer retrieval algorithms. In simple terms, the combined algorithms use the radiometer signal as a constraint on the attenuation seen by the radar.
The combined retrievals produce a hydrometeor profile, particle size distribution and surface parameters for which brightness temperatures and reflectivities are consistent with the actual satellite measuremen
For the theoretical and mathematical details of the combined radar/radiometer algorithm please review the GPM Combined Radar-Radiometer Precipitation Algorithm Theoretical Basis Document (ATBD)
https://gpm.nasa.gov/resources/documents/gpm-combined-radar-radiometer-precipitation-algorithm-theoretical-basis


VIS/IR precipitation


IMERG

The Integrated Multi-satellite Retrievals for GPM (IMERG) algorithm is run three times,
first 4 hours after the observation time (IMERG Early), then after 14 hours (IMERG Late), and finally 3.5 months later (IMERG Final).

The GEO-IR brightness temperatures used are from the Climate Prediction Center (CPC) Merged 4-km Global IR data product, which gives the "best" GEO-IR value each half hour in each ~4x4 km (at the Equator) grid box.

 Estimates based on thermal infrared sensors are lower quality due to the indirect relationship between infrared and precipitation, but they provide much more frequent coverage due to the sensors' position in geosynchronous orbit.
after all data including monthly rain gauge data are received

The most frequent number of samples of PMW data for an IMERG grid box in any given 30-minute period is overwhelmingly zero, with one in most of the rest.
Values from these grid box are derived from  "morphed" passive microwave values from previous and subsequent timesteps and/or GEO-IR derived precipitation estimates.
IMERG Early only has forward propagation (spatial extrapolation forward in time), while the Late has both forward and backward propagation (allowing interpolation in space-time)

In the IMERG-FR product, monthly precipitation gauge data are used to correct for biases that satellite data sets can exhibits.

Treat IMERG data as the average precipitation rate for the half-hour period

Additional information for IMERG is provided in the GPM FAQ or in the theoretical and mathematical details of the multi-satellite algorithm please review the ATBD https://gpm.nasa.gov/resources/documents/algorithm-information/IMERG-V06-ATBD
