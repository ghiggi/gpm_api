#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  3 14:50:39 2020

@author: ghiggi
"""
import pandas as pd
import numpy as np 
import xarray as xr
import dask.array
import h5py

from .io import find_GPM_files
from .io import GPM_IMERG_available
from .io import GPM_products_available
from .utils.utils_HDF5 import hdf5_file_attrs
from .utils.utils_string import str_remove

def subset_dict(x, keys):
    return dict((k, x[k]) for k in keys)

def remove_dict_keys(x, keys):
    if (isinstance(keys,str)):
        keys = [keys]
    for key in keys:
        x.pop(key, None)
    return
 
def flip_boolean(x):
    # return list(~numpy.array(x))
    # return [not i for i in x]
    # return list(np.invert(x))
    return list(np.logical_not(x))
    
    
def parse_GPM_ScanTime(h):    
    df = pd.DataFrame({'year': h['Year'][:],
                       'month': h['Month'][:],
                       'day': h['DayOfMonth'][:],
                       'hour': h['Hour'][:],
                       'minute': h['Minute'][:],
                       'second': h['Second'][:]})
    return pd.to_datetime(df).to_numpy()
#----------------------------------------------------------------------------.
################################
### GPM variables dictionary ###
################################
def GPM_1B_variables_dict():
    """Return a dictionary with 1B-Ku/Ka variables information."""
    dict_var = {
        'echoPower': {'path': 'Receiver/echoPower', 
                      'description':  
                          ''' The total signal power that includes both echo and noise power.
                              The range is -120 dBm to -20 dBm
                          ''',
                      'ndims': 3, 
                      'dims': ['along_track', 'cross_track', 'range'],
                      '_FillValue': [-29999, -30000],
                      'offset_scale': [0, 100],
                      'units': '',
                      'standard_name': 'echoPower'},            
        'noisePower': {'path': 'Receiver/noisePower', 
                       'description': 
                           '''An average of the received noise power for each ray
                              The range is -120 dBm to -20 dBm.
                           ''',
                       'ndims': 2, 
                       'dims': ['along_track', 'cross_track'],
                       '_FillValue': -30000,
                       'offset_scale': [0, 100],
                       'units': '',
                       'standard_name': 'noisePower'},
        'scanAngle': {'path': 'rayPointing/scanAngle', 
                      'description':  
                          ''' Angle (degrees) of the ray from nominal nadir offset 
                              about the mechanical x axis.
                              The angle is positive to the right of the direction of travel
                              Values range from -18 to 18 degrees
                          ''',
                      'ndims': 2, 
                      'dims': ['along_track', 'cross_track'],
                      '_FillValue': -9999.9,
                      'offset_scale': [],
                      'units': '',
                      'standard_name': 'scanAngle'}, 
        'localZenithAngle': {'path': 'VertLocate/scLocalZenith', 
                          'description':  
                              ''' Angle (degrees) between the local zenith and
                                  the beam’s center line.
                                  Values range from 0 to 90 degrees
                              ''',
                          'ndims': 2, 
                          'dims': ['along_track', 'cross_track'],
                          '_FillValue': -9999.9,
                          'offset_scale': [],
                          'units': '',
                          'standard_name': 'scLocalZenith'},  
        'binDiffPeakDEM': {'path': 'HouseKeeping/binDiffPeakDEM', 
                       'description': 
                           ''' The number of range bins between binEchoPeak and binDEM.
                               It is used to ensure that the VPRF is switched in 
                               accordance with the GPM satellite altitude.
                               Values range from -260 to 260 range bin number at NS and MS 
                               Values range from -130 to 130 range bin number at HS.
                           ''',
                        'ndims': 2, 
                        'dims': ['along_track', 'cross_track'],
                        '_FillValue': -9999,
                        'offset_scale': [],
                        'units': '',
                        'standard_name': 'binDiffPeakDEM'},  
        'binEchoPeak': {'path': 'VertLocate/binEchoPeak', 
                       'description': 
                           ''' The range bin number which has maximum echoPower.  
                               Values range from 1 to 260 range bin number.
                           ''',
                        'ndims': 2, 
                        'dims': ['along_track', 'cross_track'],
                        '_FillValue': -9999,
                        'offset_scale': [],
                        'units': '',
                        'standard_name': 'binEchoPeak'},              
        'alongTrackBeamWidth': {'path': 'VertLocate/alongTrackBeamWidth', 
                       'description': 
                           ''' Radar beamwidth (degrees) at the point transmitted power 
                               reaches one half of peak power in the along-track direction.
                           ''',
                        'ndims': 2, 
                        'dims': ['along_track', 'cross_track'],
                        '_FillValue': [],
                        'offset_scale': [],
                        'units': '',
                        'standard_name': 'alongTrackBeamWidth'},      
        'crossTrackBeamWidth': {'path': 'VertLocate/crossTrackBeamWidth', 
                       'description': 
                           ''' Radar beamwidth (degrees) at the point transmitted power 
                               reaches one half of peak power in the cross-track direction.
                           ''',
                        'ndims': 2, 
                        'dims': ['along_track', 'cross_track'],
                        '_FillValue': [],
                        'offset_scale': [],
                        'units': '',
                        'standard_name': 'crossTrackBeamWidth'}, 
        'mainlobeEdge': {'path': 'VertLocate/mainlobeEdge', 
                         'description': 
                           ''' Absolute value of the difference in Range Bin Numbers
                               between the detected surface and the edge of the clutter
                               from the mainlobe.
                           ''',
                         'ndims': 2, 
                         'dims': ['along_track', 'cross_track'],
                         '_FillValue': [],
                         'offset_scale': [],
                         'units': '',
                         'standard_name': 'mainlobeEdge'},
        'sidelobeRange': {'path': 'VertLocate/sidelobeRange', 
                          'description': 
                           ''' Absolute value of the difference in Range Bin Numbers
                               between the detected surface and the clutter position 
                               from the sidelobe of the clutter.
                               A zero means no clutter. 
                           ''',
                           'ndims': 2, 
                           'dims': ['along_track', 'cross_track'],
                           '_FillValue': [],
                           'offset_scale': [],
                           'units': '',
                           'standard_name': 'sidelobeRange'},
        'landOceanFlag': {'path': 'VertLocate/landOceanFlag', 
                       'description': 
                           '''Land or ocean information.
                              0=Water, 1=Land, 2=Coast, 3=Inland Water
                           ''',
                        'ndims': 2, 
                        'dims': ['along_track', 'cross_track'],
                        '_FillValue': -9999,
                        'offset_scale': [],
                        'units': '',
                        'standard_name': 'landOceanFlag'},   
        'DEMHmean': {'path': 'VertLocate/DEMHmean', 
                       'description': 
                           ''' Averaged DEM (SRTM-30) height.   
                           ''',
                        'ndims': 2, 
                        'dims': ['along_track', 'cross_track'],
                        '_FillValue': -9999,
                        'offset_scale': [],
                        'units': 'm',
                        'standard_name': 'DEMHmean'}, 
        'binDEMHbottom': {'path': 'VertLocate/binDEMHbottom', 
                       'description': 
                           ''' The range bin number of the minimum DEM elevation 
                              in a 5x5 km box centered on the IFOV.
                           ''',
                        'ndims': 2, 
                        'dims': ['along_track', 'cross_track'],
                        '_FillValue': -9999,
                        'offset_scale': [],
                        'units': '',
                        'standard_name': 'binDEMHbottom'},      
        'binDEMHtop': {'path': 'VertLocate/binDEMHtop', 
                       'description': 
                           ''' The range bin number of the maximum DEM elevation 
                               in a 5x5 km box centered on the IFOV.
                           ''',
                        'ndims': 2, 
                        'dims': ['along_track', 'cross_track'],
                        '_FillValue': -9999,
                        'offset_scale': [],
                        'units': '',
                        'standard_name': 'binDEMHtop'},      
        'binDEM': {'path': 'VertLocate/binDEM', 
                   'description': 
                           ''' The range bin number of the average DEM elevation.  
                               in a 5x5 km box centered on the IFOV.
                           ''',
                   'ndims': 2, 
                   'dims': ['along_track', 'cross_track'],
                   '_FillValue': -9999,
                   'offset_scale': [],
                   'units': '',
                   'standard_name': 'binDEM'} 
    } # close dictionary here 
    return dict_var    

##---------------------------------------------------------------------------.
def GPM_2A_variables_dict():   
    """Return a dictionary with 2A-Ku/Ka/DPR variables information."""
    # dBZm’ (zFactorMeasured) = dBZm - adjustFactor
    # dBs0m’ (sigmaZeroMeasured) = dBs0m - adjustFactor
    dict_var = {
        ### 3D data
        'precipRate': {'path': 'SLV/precipRate', 
                       'description': 'Precipitation rate.',
                       'ndims': 3, 
                       'dims': ['along_track', 'cross_track','range'],
                       '_FillValue': -9999.9,
                       'offset_scale': [],
                       'units': 'mm/hr',
                       'standard_name': 'precipRate'},
        'zFactorMeasured': {'path': 'PRE/zFactorMeasured', 
                            'description':  
                                ''' Vertical profile of reflectivity 
                                    without attenuation correction.
                                ''',
                            'ndims': 3, 
                            'dims': ['along_track', 'cross_track','range'],
                            '_FillValue': -9999.9,
                            'offset_scale': [],
                            'units': 'dBZ',
                            'standard_name': 'zFactorMeasured'},
        'zFactorCorrected': {'path': 'SLV/zFactorCorrected', 
                            'description':  
                                ''' Vertical profile of reflectivity with 
                                    attenuation correctionattenuation correction.
                                ''',
                            'ndims': 3, 
                            'dims': ['along_track', 'cross_track','range'],
                            '_FillValue': -9999.9,
                            'offset_scale': [],
                            'units': 'dBZ',
                            'standard_name': 'zFactorCorrected'}, 
        'attenuationNP': {'path': 'VER/attenuationNP', 
                          'description':  
                                ''' Vertical profile of attenuation by non-precipitation particles.
                                    (cloud liquid water, cloud ice water, water vapor, and oxygen molecules)
                                ''',
                          'ndims': 3, 
                          'dims': ['along_track', 'cross_track','range'],
                          '_FillValue': -9999.9,
                          'offset_scale': [],
                          'units': 'dB/km',
                          'standard_name': 'attenuationNP'},  
        'paramDSD': {'path': 'SLV/paramDSD', 
                     'description': 
                         ''' Parameters of the drop size distribution.''',
                     'ndims': 4, 
                     'dims': ['along_track', 'cross_track','range','nDSD'],
                     '_FillValue': -9999.9,
                     'offset_scale': [],
                     'units': '',
                     'standard_name': 'paramDSD'},  
        'phase': {'path': 'DSD/phase', 
                  'description':  
                      ''' Phase state of the precipitation. 
                            To be decoded following GPM ATBD.
                      ''',
                  'ndims': 3, 
                  'dims': ['along_track', 'cross_track','range'],
                  '_FillValue': 255,
                  'offset_scale': [],
                  'units': '',
                  'standard_name': 'phase'},     
         ### 2D data
        'zFactorCorrectedESurface': {'path': 'SLV/zFactorCorrectedESurface', 
                      'description':  
                          ''' Reflectivity factor with attenuation correction 
                              at the estimated surface
                          ''',
                      'ndims': 2, 
                      'dims': ['along_track', 'cross_track'],
                      '_FillValue': -9999.9,
                      'offset_scale': [],
                      'units': 'dBZ',
                      'standard_name': 'zFactorCorrectedESurface'}, 
         'zFactorCorrectedNearSurface': {'path': 'SLV/zFactorCorrectedNearSurface', 
                      'description':  
                          ''' Reflectivity factor with attenuation correction 
                              at the near surface
                          ''',
                      'ndims': 2, 
                      'dims': ['along_track', 'cross_track'],
                      '_FillValue': -9999.9,
                      'offset_scale': [],
                      'units': 'dBZ',
                      'standard_name': 'zFactorCorrectedNearSurface'},   
        'precipRateNearSurface': {'path': 'SLV/precipRateNearSurface', 
                      'description':  
                          ''' Precipitation rate at the near surface.
                          ''',
                      'ndims': 2, 
                      'dims': ['along_track', 'cross_track'],
                      '_FillValue': -9999.9,
                      'offset_scale': [],
                      'units': 'mm/hr',
                      'standard_name': 'precipRateNearSurface'}, 
        'precipRateESurface': {'path': 'SLV/precipRateESurface', 
                      'description':  
                          ''' Precipitation rate at the estimated surface.
                          ''',
                      'ndims': 2, 
                      'dims': ['along_track', 'cross_track'],
                      '_FillValue': -9999.9,
                      'offset_scale': [],
                      'units': 'mm/hr',
                      'standard_name': 'precipRateESurface'}, 
         'precipRateESurface2': {'path': 'Experimental/precipRateESurface2', 
                      'description':  
                          ''' Estimates Surface Precipitation using alternate method.
                          ''',
                      'ndims': 2, 
                      'dims': ['along_track', 'cross_track'],
                      '_FillValue': -9999.9,
                      'offset_scale': [],
                      'units': 'mm/hr',
                      'standard_name': 'precipRateESurface2'}, 
         'precipRateAve24': {'path': 'SLV/precipRateAve24', 
                      'description':  
                          ''' Average of precipitation rate between 2 and 4km height.
                          ''',
                      'ndims': 2, 
                      'dims': ['along_track', 'cross_track'],
                      '_FillValue': -9999.9,
                      'offset_scale': [],
                      'units': 'mm/hr',
                      'standard_name': 'precipRateAve24'},    
        'precipWaterIntegrated': {'path': 'SLV/precipWaterIntegrated', 
                 'description':  
                     ''' Precipitation water vertically integrated.
                     ''',
                 'ndims': 3, 
                 'dims': ['along_track', 'cross_track',"LS"],
                 '_FillValue': -9999.9,
                 'offset_scale': [],
                 'units': 'mm/hr',
                 'standard_name': 'precipWaterIntegrated'},    
        'seaIceConcentration': {'path': 'Experimental/seaIceConcentration', 
                      'description':  
                          ''' Sea ice concentration estimated by Ku.
                              Values range from 30 to 100 percent.
                          ''',
                      'ndims': 2, 
                      'dims': ['along_track', 'cross_track'],
                      '_FillValue': -9999.9,
                      'offset_scale': [],
                      'units': '%',
                      'standard_name': 'seaIceConcentration'}, 
        'sigmaZeroMeasured': {'path': 'PRE/sigmaZeroMeasured', 
                      'description':  
                          ''' Surface backscattering cross section without attenuation correction.
                          ''',
                      'ndims': 2, 
                      'dims': ['along_track', 'cross_track'],
                      '_FillValue': -9999.9,
                      'offset_scale': [],
                      'units': 'dB',
                      'standard_name': 'sigmaZeroMeasured'},  
         'sigmaZeroNPCorrected': {'path': 'VER/sigmaZeroNPCorrected', 
                      'description':  
                          ''' Surface backscattering cross section with 
                          attenuation correction only for non-precipitation particles
                          ''',
                      'ndims': 2, 
                      'dims': ['along_track', 'cross_track'],
                      '_FillValue': -9999.9,
                      'offset_scale': [],
                      'units': 'dB',
                      'standard_name': 'sigmaZeroNPCorrected'},   
         'sigmaZeroCorrected': {'path': 'SLV/sigmaZeroCorrected', 
                      'description':  
                          ''' Surface backscattering cross section with
                          attenuation correction 
                          ''',
                      'ndims': 2, 
                      'dims': ['along_track', 'cross_track'],
                      '_FillValue': -9999.9,
                      'offset_scale': [],
                      'units': 'dB',
                      'standard_name': 'sigmaZeroCorrected'},    
         'snRatioAtRealSurface': {'path': 'PRE/snRatioAtRealSurface', 
                      'description':  
                          ''' Signal/Noise ratio at real surface range bin
                          ''',
                      'ndims': 2, 
                      'dims': ['along_track', 'cross_track'],
                      '_FillValue': -9999,
                      'offset_scale': [],
                      'units': '',
                      'standard_name': 'snRatioAtRealSurface'},   
        'PIA': {'path': 'SRT/pathAtten', 
                'description': 'The effective 2-way path integrated attenuation.',
                'ndims': 2, 
                'dims': ['along_track', 'cross_track'],
                '_FillValue': -9999.9,
                'offset_scale': [],
                'units': 'dB',
                'standard_name': 'pathAtten'}, 
        'PIA_HB': {'path': 'SRT/PIAhb', 
                'description': 'The  2-way path integrated attenuation of HB.',
                'ndims': 2, 
                'dims': ['along_track', 'cross_track'],
                '_FillValue': -9999.9,
                'offset_scale': [],
                'units': 'dB',
                'standard_name': 'PIAhb'}, 
        'PIA_Hybrid': {'path': 'SRT/PIAhybrid', 
                'description':
                    ''' The 2-way PIA from a weighted combination of HB and SRT''',
                'ndims': 2, 
                'dims': ['along_track', 'cross_track'],
                '_FillValue': -9999.9,
                'offset_scale': [],
                'units': 'dB',
                'standard_name': 'PIAhybrid'}, 
        'PIA_Final': {'path': 'SLV/piaFinal', 
                'description': 
                    ''' The final estimates of path integrated attenuation 
                        caused by precipitation particles.
                    ''',
                'ndims': 2, 
                'dims': ['along_track', 'cross_track'],
                '_FillValue': -9999.9,
                'offset_scale': [],
                'units': 'dB',
                'standard_name': 'piaFinal'},               
        'flagPrecip': {'path': 'PRE/flagPrecip', 
                      'description':  
                          ''' Flag for precipitation. 
                          0 : No precipitation
                          1 : Precipitation
                          ''',
                      'ndims': 2, 
                      'dims': ['along_track', 'cross_track'],
                      '_FillValue': -9999,
                      'offset_scale': [],
                      'units': '',
                      'standard_name': 'flagPrecip'}, 
        'flagShallowRain': {'path': 'CSF/flagShallowRain', 
                     'description':  
                         ''' Flag for shallow precipitation: 
                             0 : No shallow rain
                             10: Shallow isolated (maybe)
                             11: Shallow isolated (certain)
                             20: Shallow non-isolated (maybe)
                             21: Shallow non-isolated (certain)
                         ''',
                     'ndims': 2, 
                     'dims': ['along_track', 'cross_track'],
                     '_FillValue': [-1111,-9999],
                     'offset_scale': [],
                     'units': '',
                     'standard_name': 'flagShallowRain'},  
        'flagBB': {'path': 'CSF/flagBB', 
                      'description':  
                          ''' Flag for Bright Band. 
                          0 : BB not detected
                          1 : BB detected
                          ''',
                      'ndims': 2, 
                      'dims': ['along_track', 'cross_track'],
                      '_FillValue': [-1111,-9999],
                      'offset_scale': [],
                      'units': '',
                      'standard_name': 'flagBB'},   
        'flagHeavyIcePrecip': {'path': 'CSF/flagHeavyIcePrecip', 
                     'description':  
                         ''' Flag for heavy precip based on reflectivity thresholds.
                             See GPM ATBD for more information.
                         ''',
                     'ndims': 2, 
                     'dims': ['along_track', 'cross_track'],
                     '_FillValue': -99,
                     'offset_scale': [],  
                     'units': '',
                     'standard_name': 'flagHeavyIcePrecip'},  
        'flagAnvil': {'path': 'CSF/flagShallowRain', 
                     'description':  
                         ''' Flag for anvil (only for Ku and DPR): 
                             0: No anvil
                             1: Anvil  
                         ''',
                     'ndims': 2, 
                     'dims': ['along_track', 'cross_track'],
                     '_FillValue': -99,
                     'offset_scale': [],
                     'units': '',
                     'standard_name': 'flagAnvil'},  
        'typePrecip': {'path': 'CSF/typePrecip', 
                      'description':  
                          ''' Precipitation type. 
                          8 digits to be decoded following GPM ATBD 
                          ''',
                      'ndims': 2, 
                      'dims': ['along_track', 'cross_track'],
                      '_FillValue': [-1111,-9999],
                      'offset_scale': [],
                      'units': '',
                      'standard_name': 'typePrecip'},   
        'flagSigmaZeroSaturation': {'path': 'PRE/flagSigmaZeroSaturation', 
                      'description':  
                          ''' Flag to show whether echoPower is under a saturated level.
                          0 : Under saturated lvel
                          1 : Possible saturated level at real surface
                          2 : Saturated level at real surface
                          ''',
                      'ndims': 2, 
                      'dims': ['along_track', 'cross_track'],
                      '_FillValue': 99,
                      'offset_scale': [],
                      'units': '',
                      'standard_name': 'flagSigmaZeroSaturation'},
         # Bin info
        'binRealSurface': {'path': 'PRE/binRealSurface', 
                      'description':  
                          ''' Range bin number for real surface.
                          ''',
                      'ndims': 2, 
                      'dims': ['along_track', 'cross_track'],
                      '_FillValue': -9999,
                      'offset_scale': [],
                      'units': '',
                      'standard_name': 'binRealSurface'},
        'binEchoBottom': {'path': 'SLV/binEchoBottom', 
                      'description':  
                          ''' Range bin number for ... (TODO)
                          ''',
                      'ndims': 2, 
                      'dims': ['along_track', 'cross_track'],
                      '_FillValue': -9999,
                      'offset_scale': [],
                      'units': '',
                      'standard_name': 'binEchoBottom'},  
        'binClutterFreeBottom': {'path': 'PRE/binClutterFreeBottom', 
                          'description':  
                              ''' Range bin number for clutter free bottom
                              ''',
                          'ndims': 2, 
                          'dims': ['along_track', 'cross_track'],
                          '_FillValue': -9999,
                          'offset_scale': [],
                          'units': '',
                          'standard_name': 'binClutterFreeBottom'},
        'binZeroDeg': {'path': 'VER/binZeroDeg', 
                  'description':  
                      ''' Range bin number with 0 °C.
                          For NS and MS: 177 --> 0 °C at the surface.
                          For HS: 89 --> 0 °C at the surface.
                      ''',
                  'ndims': 2, 
                  'dims': ['along_track', 'cross_track'],
                  '_FillValue': [],
                  'offset_scale': [],
                  'units': '',
                  'standard_name': 'binZeroDeg'}, 
        'binBBBottom': {'path': 'CSF/binBBBottom', 
                  'description':  
                      ''' Range bin number for the bottom of the bright band
                      ''',
                  'ndims': 2, 
                  'dims': ['along_track', 'cross_track'],
                  '_FillValue': [-1111,-9999],
                  'offset_scale': [],
                  'units': '',
                  'standard_name': 'binBBBottom'}, 
        'binBBPeak': {'path': 'CSF/binBBPeak', 
                  'description':  
                      ''' Range bin number for the peak of the bright band
                      ''',
                  'ndims': 2, 
                  'dims': ['along_track', 'cross_track'],
                  '_FillValue': [-1111,-9999],
                  'offset_scale': [],
                  'units': '',
                  'standard_name': 'binBBPeak'}, 
        'binBBTop': {'path': 'CSF/binBBTop', 
                  'description':  
                      ''' Range bin number for the top of the bright band
                      ''',
                  'ndims': 2, 
                  'dims': ['along_track', 'cross_track'],
                  '_FillValue': [-1111,-9999],
                  'offset_scale': [],
                  'units': '',
                  'standard_name': 'binBBTop'}, 
        'binDFRmMLBottom': {'path': 'CSF/binDFRmMLBottom', 
                            'description':  
                              ''' Range bin number for the for melting layer bottom''',                      
                            'ndims': 2, 
                            'dims': ['along_track', 'cross_track'],
                            '_FillValue': [0,-1111,-9999],
                            'offset_scale': [],
                            'units': '',
                            'standard_name': 'binDFRmMLBottom'}, 
        'binDFRmMLTop': {'path': 'CSF/binDFRmMLTop', 
                         'description':  
                              ''' Range bin number for the for melting layer top''',                      
                         'ndims': 2, 
                         'dims': ['along_track', 'cross_track'],
                         '_FillValue': [0,-1111,-9999],
                         'offset_scale': [],
                         'units': '',
                         'standard_name': 'binDFRmMLTop'},     
        'binStormTop': {'path': 'PRE/binStormTop', 
                  'description':  
                      ''' Range bin number for the storm top.
                      ''',
                  'ndims': 2, 
                  'dims': ['along_track', 'cross_track'],
                  '_FillValue': -9999,
                  'offset_scale': [],
                  'units': '',
                  'standard_name': 'binStormTop'},  
        # Heights info 
        'heightZeroDeg': {'path': 'VER/heightZeroDeg', 
                          'description':  
                              ''' Height of freezing level (0 degrees C level).''',
                          'ndims': 2, 
                          'dims': ['along_track', 'cross_track'],
                          '_FillValue': -9999.9,
                          'offset_scale': [],
                          'units': 'm',
                          'standard_name': 'heightZeroDeg'},  
        'heightBB': {'path': 'CSF/heightBB', 
                          'description':  
                              ''' Height of the bright band. ''',
                          'ndims': 2, 
                          'dims': ['along_track', 'cross_track'],
                          '_FillValue': [-1111.1,-9999.9],
                          'offset_scale': [],
                          'units': 'm',
                          'standard_name': 'heightBB'},     
        'widthBB': {'path': 'CSF/widthBB', 
                          'description':  
                              ''' Width of the bright band. ''',
                          'ndims': 2, 
                          'dims': ['along_track', 'cross_track'],
                          '_FillValue': [-1111.1,-9999.9],
                          'offset_scale': [],
                          'units': 'm',
                          'standard_name': 'widthBB'},       
        'heightStormTop': {'path': 'PRE/heightStormTop', 
                           'description':  
                               ''' Height of storm top. ''',
                           'ndims': 2, 
                           'dims': ['along_track', 'cross_track'],
                           '_FillValue': -9999.9,
                           'offset_scale': [],
                           'units': 'm',
                           'standard_name': 'heightStormTop'},  
        'elevation': {'path': 'PRE/elevation', 
                      'description':  
                          ''' Elevation of the measurement point. 
                              It is a copy of DEMHmean of level 1B product.
                          ''',
                      'ndims': 2, 
                      'dims': ['along_track', 'cross_track'],
                      '_FillValue': -9999.9,
                      'offset_scale': [],
                      'units': 'm',
                      'standard_name': 'elevation'}, 
        'localZenithAngle': {'path': 'PRE/localZenithAngle', 
                      'description':  
                          ''' Local zenith angle of each ray (scLocalZenith)
                          ''',
                      'ndims': 2, 
                      'dims': ['along_track', 'cross_track'],
                      '_FillValue': -9999.9,
                      'offset_scale': [],
                      'units': 'degree',
                      'standard_name': 'localZenithAngle'},     
            
            
        # Quality Flags 
        'qualityFlag': {'path': 'FLG/qualityFlag', 
                          'description':  
                              ''' Quality flag: 
                                  0 : High quality. No issues
                                  1 : Low quality (DPR modules had warnings but still made a retrieval)
                                  2 : Bad (DPR modules had errors)
                              ''',
                          'ndims': 2, 
                          'dims': ['along_track', 'cross_track'],
                          '_FillValue': [-1111,-9999],
                          'offset_scale': [],
                          'units': 'm',
                          'standard_name': 'qualityFlag'},
        'qualityBB': {'path': 'CSF/qualityBB', 
                          'description':  
                              ''' Quality of the bright band detection: 
                                  0 : BB not detected in the case of rain
                                  1 : Clear bright band
                                  2 : Not so clear bright band
                                  3 : Smeared bright band
                              ''',
                          'ndims': 2, 
                          'dims': ['along_track', 'cross_track'],
                          '_FillValue': [-1111,-9999],
                          'offset_scale': [],
                          'units': 'm',
                          'standard_name': 'qualityBB'},
        'qualityPIA': {'path': 'SRT/reliabFlag', 
                       'description':  
                             '''The reliability flag for the effective PIA estimate (pathAtten).
                             1 : Reliable
                             2 : Marginally reliable 
                             3 : Unreliable
                             4 : Provides a lower bound to the path-attenuation
                             ''',
                       'ndims': 2, 
                       'dims': ['along_track', 'cross_track'],
                       '_FillValue': 9,
                       'offset_scale': [],
                       'units': 'dB',
                       'standard_name': 'reliabFlag'},          
    } # close dictionary here 
    return dict_var

##----------------------------------------------------------------------------.
def GPM_2A_ENV_variables_dict(): 
    """Return a dictionary with 2A-ENV variables information."""
    dict_var = {
    'airPressure': {'path': 'VERENV/airPressure', 
                    'description': 'Air Pressure',
                    'ndims': 3, 
                    'dims': ['along_track','cross_track', 'range'],
                    '_FillValue': -9999.9,
                    'offset_scale': [],
                    'units': 'hPa',
                    'standard_name': 'airPressure'}, 
    'airTemperature': {'path': 'VERENV/airTemperature', 
                    'description': 'Air Temperature',
                    'ndims': 3, 
                    'dims': ['along_track','cross_track', 'range'],
                    '_FillValue': -9999.9,
                    'offset_scale': [],
                    'units': 'K',
                    'standard_name': 'airTemperature'}, 
    'cloudLiquidWater': {'path': 'VERENV/cloudLiquidWater', 
                    'description': 'Cloud Liquid Water',
                    'ndims': 3, 
                    'dims': ['along_track','cross_track', 'range', 'nwater'],
                    '_FillValue': -9999.9,
                    'offset_scale': [],
                    'units': 'kg/m^3',
                    'standard_name': 'cloudLiquidWater'}, 
    'waterVapor': {'path': 'VERENV/waterVapor', 
                   'description': 'waterVapor',
                   'ndims': 4, 
                   'dims': ['along_track','cross_track','range','nwater'],
                   '_FillValue': -9999.9,
                   'offset_scale': [],
                   'units': 'kg/m^3',
                   'standard_name': 'waterVapor'}, 
    }
    return dict_var

##----------------------------------------------------------------------------.
def GPM_2A_SLH_variables_dict(): 
    """Return a dictionary with 2A-SLH variables information."""
    dict_var = {
    'latentHeating': {'path': 'latentHeating', 
                      'description': 'Latent Heating. Values range from -400 to 400 K/hr.',
                      'ndims': 3, 
                      'dims': ['along_track','cross_track', 'layer'],
                      '_FillValue': -9999.9,
                      'offset_scale': [],
                      'units': 'K/hr',
                      'standard_name': 'latentHeating'},  
    'meltLayerHeight': {'path': 'meltLayerHeight', 
                        'description': 'Height of melting layer. Values range from 0 to 32000 m.',
                        'ndims': 2, 
                        'dims': ['along_track','cross_track'],
                        '_FillValue': -9999,
                        'offset_scale': [],
                        'units': 'm',
                        'standard_name': 'meltLayerHeight'},
    'stormTopHeight': {'path': 'stormTopHeight', 
                    'description': 'Height of storm top. Values range from 0 to 32000 m.',
                    'ndims': 2, 
                    'dims': ['along_track','cross_track'],
                    '_FillValue': -9999,
                    'offset_scale': [],
                    'units': 'm',
                    'standard_name': 'stormTopHeight'},
    'rainTypeSLH': {'path': 'rainTypeSLH', 
                    'description': 'Rain type of SLH. See GPM ATBD.',
                    'ndims': 2, 
                    'dims': ['along_track','cross_track'],
                    '_FillValue': -9999,
                    'offset_scale': [],
                    'units': '',
                    'standard_name': 'latentHeating'},
    'nearSurfHeight': {'path': 'nearSurfLevel', 
                       'description': 'Height of the near surface level.',
                       'ndims': 2, 
                       'dims': ['along_track','cross_track'],
                       '_FillValue': -9999,
                       'offset_scale': [],
                       'units': 'm',
                       'standard_name': 'nearSurfLevel'},
    'topoHeight': {'path': 'topoLevel', 
                   'description': 'Height of the topography',
                   'ndims': 2, 
                   'dims': ['along_track','cross_track'],
                   '_FillValue': -9999,
                   'offset_scale': [],
                   'units': 'm',
                   'standard_name': 'topoLevel'},
    'climMeltLayerHeight': {'path': 'climMeltLevel', 
                            'description': 'Climatological height of the melting layer.',
                            'ndims': 2, 
                            'dims': ['along_track','cross_track'],
                            '_FillValue': -9999,
                            'offset_scale': [],
                            'units': 'm',
                            'standard_name': 'climMeltLevel'},
    'climFreezLayerHeight': {'path': 'climFreezLevel', 
                             'description': 'Climatological height of the freezing layer.',
                             'ndims': 2, 
                             'dims': ['along_track','cross_track'],
                             '_FillValue': -9999,
                             'offset_scale': [],
                             'units': 'm',
                             'standard_name': 'climFreezLevel'},
    } # close dictionary here 
    return dict_var

##----------------------------------------------------------------------------.
def GPM_IMERG_variables_dict():   
    """Return a dictionary with IMERG variables information."""
    dict_var = {
    'precipitationCal': {'path': 'precipitationCal', 
                         'description': 'Precipitation estimate using gauge calibration over land.',
                         'ndims': 3, 
                         'dims': ['time','lon', 'lat'],
                         '_FillValue': -9999.9,
                         'offset_scale': [],
                         'units': 'mm/hr',
                         'standard_name': 'precipitationCal'},   
    'precipitationUncal': {'path': 'precipitationUncal', 
                           'description': 'Precipitation estimate with no gauge calibration.',
                           'ndims': 3, 
                           'dims': ['time','lon', 'lat'],
                           '_FillValue': -9999.9,
                           'offset_scale': [],
                           'units': 'mm/hr',
                           'standard_name': 'precipitationUncal'},   
    'randomError': {'path': 'randomError', 
                    'description': 'Random error estimate of precipitation.',
                    'ndims': 3, 
                    'dims': ['time','lon', 'lat'],
                    '_FillValue': -9999.9,
                    'offset_scale': [],
                    'units': 'mm/hr',
                    'standard_name': 'randomError'},
    'probabilityLiquidPrecipitation': {'path': 'probabilityLiquidPrecipitation', 
                                       'description': 
                                           """ Probability of liquid precipitation
                                           0=definitely frozen. 
                                           100=definitely liquid.
                                           50=equal probability frozen or liquid.
                                          """,
                                       'ndims': 3, 
                                       'dims': ['time','lon', 'lat'],
                                       '_FillValue': -9999,
                                       'offset_scale': [],
                                       'units': '',
                                       'standard_name': 'probabilityLiquidPrecipitation'},
    'IRprecipitation': {'path': 'IRprecipitation', 
                        'description': 'Microwave-calibrated IR precipitation estimate.',
                        'ndims': 3, 
                        'dims': ['time','lon', 'lat'],
                        '_FillValue': -9999.9,
                        'offset_scale': [],
                        'units': 'mm/hr',
                        'standard_name': 'IRprecipitation'}, 
    'PMWprecipitation': {'path': 'HQprecipitation', 
                        'description': 'Instananeous microwave-only precipitation estimate.',
                        'ndims': 3, 
                        'dims': ['time','lon', 'lat'],
                        '_FillValue': -9999.9,
                        'offset_scale': [],
                        'units': 'mm/hr',
                        'standard_name': 'HQprecipitation'}, 
    'PMWprecipSource': {'path': 'HQprecipSource', 
                        'description': """
                            Sensor used for HQprecipitation:
                            0 = no observation
                            1 = TMI
                            2 = TCI
                            3 = AMSR
                            4 = SSMI
                            5 = SSMIS
                            6 = AMSU
                            7 = MHS
                            8 = Megha-Tropiques
                            9 = GMI
                            10 = GCI
                            11 = ATMS
                            12 = AIRS
                            13 = TOVS
                            14 = CrlS
                            """,
                       'ndims': 3, 
                       'dims': ['time','lon', 'lat'],
                       '_FillValue': -9999.9,
                       'offset_scale': [],
                       'units': 'mm/hr',
                       'standard_name': 'HQprecipSource'}, 
    'PMWobservationTime': {'path': 'HQobservationTime', 
                           'description': """
                             Observation time (from the beginnning of the current half hour) 
                             of the instantaneous PMW-only precipitation estimate 
                             covering the current 30-minute period. 
                             Values range from 0 to 29 minutes.
                             """,
                           'ndims': 3, 
                           'dims': ['time','lon', 'lat'],
                           '_FillValue': -9999,
                           'offset_scale': [],
                           'units': 'minutes',
                           'standard_name': 'HQobservationTime'}, 
    'precipitationQualityIndex': {'path': 'precipitationQualityIndex', 
                                  'description': """
                                      Estimated quality of precipitationCal where
                                      0 is worse and 100 is better. 
                                      """,
                                  'ndims': 3, 
                                  'dims': ['time','lon', 'lat'],
                                  '_FillValue': -9999,
                                  'offset_scale': [],
                                  'units': '',
                                  'standard_name': 'precipitationQualityIndex'}, 
    } # close dictionary here
    return dict_var

#----------------------------------------------------------------------------. 
def GPM_variables_dict(product): 
    """
    Return a dictionary with variables information for a specific GPM product.

    Parameters
    ----------
    product : str
        GPM product acronym.
        
    Returns
    -------
    dict

    """
    if (product in ['1B-Ku','1B-Ka']):
        variables_dict = GPM_1B_variables_dict()
    elif (product in ['2A-Ku','2A-Ka','2A-DPR']):
        variables_dict = GPM_2A_variables_dict()
    elif (product in ['2A-ENV-DPR','2A-ENV-Ka','2A-ENV-Ku']):
        variables_dict = GPM_2A_ENV_variables_dict()
    elif (product in ['2A-SLH']):
        variables_dict = GPM_2A_SLH_variables_dict()
    elif (product in GPM_IMERG_available()):
        variables_dict = GPM_IMERG_variables_dict()
    else:
        raise ValueError('What is going on?')
    return(variables_dict)

def GPM_variables(product):
    """
    Return a list of variables available for a specific GPM product.

    Parameters
    ----------
    product : str
        GPM product acronym.
        
    Returns
    -------
    list

    """
    return list(GPM_variables_dict(product).keys())

#-----------------------------------------------------------------------------.
###############
### Checks ####
###############
def check_product(product):
    """Checks the validity of product."""
    if not isinstance(product, str):
        raise ValueError('Ask for a single product at time') 
    if not (product in GPM_products_available()):
        raise ValueError('Retrieval for such product not available') 
    return 

##----------------------------------------------------------------------------.
def check_scan_mode(scan_mode, product):
    """Checks the validity of scan_mode."""
    # Check that the scan mode is specified if asking for radar data
    if ((scan_mode is None) and (product in ['1B-Ku','1B-Ka','2A-Ku','2A-Ka','2A-DPR',
                                             '2A-ENV-DPR','2A-ENV-Ka','2A-ENV-Ku'])):
        raise ValueError('Please specify a valid scan_mode: NS, MS, HS')
    # Check that a single scan mode is specified 
    if ((scan_mode is not None) and not (isinstance(scan_mode, str))):
        raise ValueError('Specify a single scan_mode at time') 
    # Check that a valid scan mode is specified if asking for radar data
    if (scan_mode is not None):
        if ((product in ['1B-Ku', '2A-Ku','2A-ENV-Ku']) and scan_mode != "NS"):
            raise ValueError("For '1B-Ku','2A-Ku'and '2A-ENV-Ku' products, specify scan_mode = 'NS'")
        if ((product in ['1B-Ka', '2A-Ka','2A-ENV-Ka']) and (scan_mode not in ['MS','HS'])):    
            raise ValueError("For '1B-Ka', '2A-Ka' and '2A-ENV-Ka' products, specify scan_mode either 'MS' or 'HS'")
        if ((product in ['2A-DPR']) and (scan_mode not in ['NS','MS','HS'])):    
            raise ValueError("For '2A-DPR' product, specify scan_mode either 'NS', 'MS' or 'HS'") 
        if ((product in ['2A-ENV-DPR']) and (scan_mode not in ['NS','HS'])):    
            raise ValueError("For '2A-ENV-DPR' products, specify scan_mode either 'NS' or 'HS'") 
    # Specify HDF group name for 2A-SLH and IMERG products
    if (product == "2A-SLH"):
        scan_mode = 'Swath'    
    if (product in ['IMERG-FR','IMERG-ER','IMERG-LR']):
        scan_mode = 'Grid'
    if (scan_mode is None):
        raise ValueError('scan_mode is still None. This should not occur!')
    return(scan_mode)  

##----------------------------------------------------------------------------.
def check_variables(variables, product, scan_mode):
    """Checks the validity of variables."""
    # Make sure variable is a list (if str --> convert to list)     
    if (isinstance(variables, str)):
        variables = [variables]
    # Check variables are valid 
    idx_valid = [var in GPM_variables(product=product) for var in variables]
    if not all(idx_valid): 
        idx_not_valid = np.logical_not(idx_valid)
        if (all(idx_not_valid)):
            raise ValueError('All variables specified are not valid')
        else:
            variables_not_valid = list(np.array(variables)[idx_not_valid])
            raise ValueError('The following variable are not valid:', variables_not_valid)
    ##------------------------------------------------------------------------.    
    # Treat special cases for variables not available for specific products
    # 1B products    
    # 2A products
    if ('flagAnvil' in variables): 
        if ((product == '2A-Ka') or (product == '2A-DPR' and scan_mode in ['MS','HS'])): 
            print('flagAnvil available only for Ku-band and DPR NS.\n Silent removal from the request done.')
            variables = str_remove(variables, 'flagAnvil')
    if ('binDFRmMLBottom' in variables):         
        if (product in ['2A-Ka','2A-Ku']):  
            print('binDFRmMLBottom available only for 2A-DPR.\n Silent removal from the request done.')
            variables = str_remove(variables, 'binDFRmMLBottom')
    if ('binDFRmMLTop' in variables):         
        if (product in ['2A-Ka','2A-Ku']):  
            print('binDFRmMLTop available only for 2A-DPR.\n Silent removal from the request done.')
            variables = str_remove(variables, 'binDFRmMLTop') 
    ##------------------------------------------------------------------------.  
    # Check that there are still some variables to retrieve
    if (len(variables) == 0):
        raise ValueError('No valid variables to retrieve')        
    return(variables)   

##----------------------------------------------------------------------------.
def check_bbox(bbox):
    """
    Check correctnes of bounding box. 
    
    bbox format: [lon_0, lon_1, lat_0, lat_1]
    bbox should be provided with longitude between -180 and 180, and latitude
    between -90 and 90.
    """
    if bbox is None:
        return bbox
    # If bbox provided 
    if not (isinstance(bbox, list) and len(bbox) == 4):
        raise ValueError('Provide valid bbox [lon_0, lon_1, lat_0, lat_1]')
    if (bbox[2] > 90 or bbox[2] < -90 or bbox[3] > 90 or bbox[3] < -90):
        raise ValueError('Latitude is defined between -90 and 90')
    # Try to be sure that longitude is specified between -180 and 180        
    if (bbox[0] > 180 or bbox[1] > 180):
        print('bbox should be provided with longitude between -180 and 180')
        bbox[0] = bbox[0] - 180      
        bbox[1] = bbox[1] - 180        
    return(bbox)

#----------------------------------------------------------------------------.     
def GPM_granule_Dataset(hdf, product, variables, 
                        scan_mode = None, 
                        variables_dict = None,
                        bbox=None, enable_dask=True, chunks='auto'):
    """
    Create a lazy xarray.Dataset with relevant GPM data and attributes 
    for a specific granule.   

    Parameters
    ----------
    hdf : h5py.File
        HFD5 object read with h5py.
    scan_mode : str
        'NS' = Normal Scan --> For Ku band and DPR 
        'MS' = Matched Scans --> For Ka band and DPR 
        'HS' = High-sensitivity Scans --> For Ka band and DPR
        For products '1B-Ku', '2A-Ku' and '2A-ENV-Ku', specify 'NS'.
        For products '1B-Ka', '2A-Ka' and '2A-ENV-Ka', specify either 'MS' or 'HS'.
        For product '2A-DPR', specify either 'NS', 'MS' or 'HS'.
        For product '2A-ENV-DPR', specify either 'NS' or 'HS'.
        For product '2A-SLH', specify scan_mode = 'Swath'.
        For product 'IMERG-ER','IMERG-LR' and 'IMERG-FR', specify scan_mode = 'Grid'.
    product : str
        GPM product acronym.                           
    variables : list, str
         Datasets names to extract from the HDF5 file.
         Hint: utils_HDF5.hdf5_datasets_names() to see available datasets.
    variables_dict : dict, optional    
         Expect dictionary from GPM_variables_dict(product)
         Provided to avoid recomputing it at every call.
         If variables_dict is None --> Perform also checks on the arguments .
    bbox : list, optional 
         Spatial bounding box. Format: [lon_0, lon_1, lat_0, lat_1]  
         For radar products it subset only along_track !
    dask : bool, optional
         Wheter to lazy load data (in parallel) with dask. The default is True.
         Hint: xarray’s lazy loading of remote or on-disk datasets is often but not always desirable.
         Before performing computationally intense operations, load the Dataset
         entirely into memory by invoking the Dataset.load() 
    chunks : str, list, optional
        Chunck size for dask. The default is 'auto'.
        Alternatively provide a list (with length equal to 'variables') specifying
        the chunk size option for each variable.
        
    Returns
    -------
    xarray.Dataset

    """   
    ##------------------------------------------------------------------------.
    ## Arguments checks are usually done in GPM _Dataset()       
    if variables_dict is None:
        ## Check valid product 
        check_product(product)
        ## Check scan_mode 
        scan_mode = check_scan_mode(scan_mode, product)      
        ## Check variables 
        variables = check_variables(variables, product, scan_mode)   
        ## Check bbox
        bbox = check_bbox(bbox)
        ##--------------------------------------------------------------------.   
        ## Retrieve variables dictionary 
        variables_dict = GPM_variables_dict(product=product)  
    ##------------------------------------------------------------------------.     
    ## Retrieve basic coordinates 
    # - For GPM radar products
    if (product not in GPM_IMERG_available()):
        lon = hdf[scan_mode]['Longitude'][:]
        lat = hdf[scan_mode]['Latitude'][:]
        tt = parse_GPM_ScanTime(hdf[scan_mode]['ScanTime'])
        coords = {'lon': (['along_track','cross_track'],lon),
                  'lat': (['along_track','cross_track'],lat),
                  'time': (['along_track'], tt)}
    # - For IMERG products
    else:
        lon = hdf[scan_mode]['lon'][:]
        lat = hdf[scan_mode]['lat'][:]
        tt = hdf5_file_attrs(hdf)['FileHeader']['StartGranuleDateTime'][:-1]  
        tt = np.array(np.datetime64(tt) + np.timedelta64(30, 'm'), ndmin=1)
        coords = {'time': tt,
                  'lon': lon,
                  'lat': lat}
    ##------------------------------------------------------------------------.
    ## Check if there is some data in the bounding box
    if (bbox is not None):
        # - For GPM radar products
        if (product not in GPM_IMERG_available()):
            idx_row, idx_col = np.where((lon >= bbox[0]) & (lon <= bbox[1]) & (lat >= bbox[2]) & (lat <= bbox[3]))              
        # - For IMERG products
        else:   
            idx_row = np.where((lon >= bbox[0]) & (lon <= bbox[1]))[0]    
            idx_col = np.where((lat >= bbox[2]) & (lat <= bbox[3]))[0]    
        # If no data in the bounding box in current granule, return empty list
        if (idx_row.size == 0 or idx_col.size == 0):
            return(None)
    ##------------------------------------------------------------------------.
    # Retrieve each variable 
    flag_first = True # Required to decide if to create/append to Dataset
    for var in variables: 
        # print(var)
        ##--------------------------------------------------------------------.
        # Prepare attributes for the DataArray 
        dict_attr = subset_dict(variables_dict[var], ['description','units','standard_name'])
        dict_attr['product'] = product
        dict_attr['scan_mode'] = scan_mode
        ##--------------------------------------------------------------------.
        # Choose if using dask 
        if enable_dask is True:
            hdf_obj = dask.array.from_array(hdf[scan_mode][variables_dict[var]['path']], chunks=chunks)
        else:
            hdf_obj = hdf[scan_mode][variables_dict[var]['path']]
        ##--------------------------------------------------------------------.    
        # Create the DataArray
        da = xr.DataArray(hdf_obj,
                          dims = variables_dict[var]['dims'],
                          coords = coords,
                          attrs = dict_attr)
        da.name = var
        ## -------------------------------------------------------------------.
        ## Subsetting based on bbox (lazy with dask)
        if bbox is not None:
            # - For GPM radar products
            # --> Subset only along_track to allow concat on cross_track  
            if (product not in GPM_IMERG_available()):
                da = da.isel(along_track = slice((min(idx_row)),(max(idx_row)+1))) 
            # - For IMERG products
            else: 
                da = da.isel(lon = idx_row, lat = idx_col)
        ## -------------------------------------------------------------------.
        ## Convert to float explicitly (needed?)
        # hdf_obj.dtype  ## int16
        # da = da.astype(np.float)
        ## -------------------------------------------------------------------.
        ## Parse missing values and errors
        da = xr.where(da.isin(variables_dict[var]['_FillValue']), np.nan, da)     
        # for value in dict_attr['_FillValue']:
        #     da = xr.where(da == value, np.nan, da)
        ## -------------------------------------------------------------------.
        ## Add scale and offset 
        if len(variables_dict[var]['offset_scale'])==2:
            da = da/variables_dict[var]['offset_scale'][1] - variables_dict[var]['offset_scale'][0]
        ## --------------------------------------------------------------------.    
        ## Create/Add to Dataset 
        if flag_first is True: 
            ds = da.to_dataset()
            flag_first = False
        else:
            ds[var] = da
        ##--------------------------------------------------------------------. 
        ## Special processing for specific fields     
        if var == 'precipWaterIntegrated':
            ds['precipWaterIntegrated_Liquid'] = ds['precipWaterIntegrated'][:,:,0]
            ds['precipWaterIntegrated_Solid'] = ds['precipWaterIntegrated'][:,:,1]
            ds['precipWaterIntegrated'] = ds['precipWaterIntegrated_Liquid']+ds['precipWaterIntegrated_Solid']
        if var == 'paramDSD':
            ds['DSD_dBNw'] = ds['paramDSD'][:,:,:,0]
            ds['DSD_m'] = ds['paramDSD'][:,:,:,1]
            ds = ds.drop_vars(names='paramDSD')
            # Modify attributes 
            ds['DSD_m'].attrs['units'] = 'mm' 
        if (var == 'flagBB' and product == '2A-DPR'):
            ds['DSD_m'].attrs['description'] = ''' Flag for Bright Band: 
                                                0 : BB not detected
                                                1 : Bright Band detected by Ku and DFRm
                                                2 : Bright Band detected by Ku only
                                                3 : Bright Band detected by DFRm only
                                               '''
        # TODO     ???                                 
        # if (var == 'cloudLiquidWater'):
        #     # nwater , 
        # if (var == 'waterVapor'):
        #     # nwater
        if (var == 'phase'):
            print('Decoding of phase not yet implemented')
        if (var == 'typePrecip'): 
            print('Decoding of typePrecip not yet implemented')    
    ##------------------------------------------------------------------------. 
    # Add optional coordinates 
    # - altitude...
    # - TODO 
    ##------------------------------------------------------------------------.
    # Add other stuffs to dataset    
    return(ds) 
      
def GPM_Dataset(base_DIR,
                product, 
                variables, 
                start_time, 
                end_time,
                scan_mode=None, 
                bbox=None, enable_dask=False, chunks='auto'):
    """
    Lazily map HDF5 data into xarray.Dataset with relevant GPM data and attributes. 
   
    Parameters
    ----------
    base_DIR : str
       The base directory where GPM data are stored.
    product : str
        GPM product acronym.                           
    variables : list, str
         Datasets names to extract from the HDF5 file.
         Hint: GPM_variables(product) to see available variables.
    start_time : datetime
        Start time.
    end_time : datetime
        End time.
    scan_mode : str, optional
        'NS' = Normal Scan --> For Ku band and DPR 
        'MS' = Matched Scans --> For Ka band and DPR 
        'HS' = High-sensitivity Scans --> For Ka band and DPR
        For products '1B-Ku', '2A-Ku' and '2A-ENV-Ku', specify 'NS'.
        For products '1B-Ka', '2A-Ka' and '2A-ENV-Ka', specify either 'MS' or 'HS'.
        For product '2A-DPR', specify either 'NS', 'MS' or 'HS'.
        For product '2A-ENV-DPR', specify either 'NS' or 'HS'.
    bbox : list, optional 
         Spatial bounding box. Format: [lon_0, lon_1, lat_0, lat_1]  
    dask : bool, optional
         Wheter to lazy load data (in parallel) with dask. The default is True.
         Hint: xarray’s lazy loading of remote or on-disk datasets is often but not always desirable.
         Before performing computationally intense operations, load the Dataset
         entirely into memory by invoking the Dataset.load() 
    chunks : str, list, optional
        Chunck size for dask. The default is 'auto'.
        Alternatively provide a list (with length equal to 'variables') specifying
        the chunk size option for each variable.

    Returns
    -------
    xarray.Dataset

    """
    ##------------------------------------------------------------------------.
    ## Check valid product 
    check_product(product)
    ## Check scan_mode 
    scan_mode = check_scan_mode(scan_mode, product)      
    ## Check variables 
    variables = check_variables(variables, product, scan_mode)   
    ## Check bbox
    bbox = check_bbox(bbox)
    ##------------------------------------------------------------------------.
    ## Check for chuncks    
    # TODO smart_autochunck per variable (based on dim...)
    # chunks = check_chuncks(chunks) 
    ##------------------------------------------------------------------------.
    # Find filepaths
    filepaths = find_GPM_files(base_DIR = base_DIR, 
                               product = product, 
                               start_time = start_time,
                               end_time = end_time)
    ##------------------------------------------------------------------------.
    # Check that files have been downloaded  on disk 
    if (len(filepaths) == 0):
        raise ValueError('Requested files are not found on disk. Please download them before')
    ##------------------------------------------------------------------------.
    # Initialize list (to store Dataset of each granule )
    l_Datasets = list() 
    # Retrieve variables dictionary 
    variables_dict = GPM_variables_dict(product=product)   
    for filepath in filepaths:  
        # Load hdf granule file  
        hdf = h5py.File(filepath,'r') # h5py._hl.files.File
        hdf_attr = hdf5_file_attrs(hdf)
        # --------------------------------------------------------------------.
        ## Decide if retrieve data based on JAXA quality flags 
        # Do not retrieve data if TotalQualityCode not ... 
        if (product not in GPM_IMERG_available()):
            DataQualityFiltering = {'TotalQualityCode': ['Good']} # TODO future fun args
            if (hdf_attr['JAXAInfo']['TotalQualityCode'] not in DataQualityFiltering['TotalQualityCode']):
                continue
        #---------------------------------------------------------------------.
        # Retrieve data if granule is not empty 
        if (hdf_attr['FileHeader']['EmptyGranule'] == 'NOT_EMPTY'):
            ds = GPM_granule_Dataset(hdf=hdf,
                                     product=product, 
                                     scan_mode = scan_mode,  
                                     variables = variables,
                                     variables_dict = variables_dict,
                                     bbox = bbox,
                                     enable_dask=enable_dask, chunks='auto')
            if ds is not None:
                l_Datasets.append(ds)
    #-------------------------------------------------------------------------.
    # Concat all Datasets
    if (len(l_Datasets) >= 1):
        if (product in GPM_IMERG_available()):
            ds = xr.concat(l_Datasets, dim="time")
        else:
            ds = xr.concat(l_Datasets, dim="along_track")
        print('GPM Dataset loaded successfully !')    
    else:
        print("No data available for current request. Try for example to modify the bbox.")
        return 
    #-------------------------------------------------------------------------.
    # Return Dataset
    return ds
