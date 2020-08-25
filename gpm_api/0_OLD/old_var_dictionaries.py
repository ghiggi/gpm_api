#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 11:09:46 2020

@author: ghiggi
"""



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


GPM_version = 6
products = GPM_products_available()
for product in products:
    scan_modes = initialize_scan_modes(product)
    for scan_mode in scan_modes:
        dict_file = GPM_variables_dict(product)
        filename = "GPM_V" + str(GPM_version) + "_" + product + "_" + scan_mode
        #filepath = '/home/ghiggi/gpm_api/gpm_api/CONFIG/' + filename + '.yaml' 
        with open(filepath, 'w') as file:
            documents = yaml.dump(dict_file, file)