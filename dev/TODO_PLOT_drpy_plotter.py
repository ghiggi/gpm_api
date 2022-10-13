#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 13:22:52 2022

@author: ghiggi
"""
# Download
io = drpy.io.netrunner(servername='Research',username='username@email.com',start_time=dtime)
# Class 
dpr = drpy.core.GPMDPR(filename=io.filename[0][-64:])

# Case study plot 
center_lat = -17.489
center_lon = 56.181
c  = drpy.graph.case_study(filename=io.filename[0][-64:],center_lat=center_lat,center_lon=center_lon)
c.plotter_along(scan=12)
c.plotter_along(start_index=130,end_index=160,scan=24)
c.plotter_along(start_index=0,end_index=-1,scan=24)
c.plotter_along(start_index=25,end_index=-25,scan=24, params_new={'y_max':15,'z_vmin':0,'z_vmax':30})
c.plotter_along(start_index=25,end_index=-25,scan=24, params_new={'y_max':15,'z_vmin':0,'z_vmax':30,
                                                                  'dfr_vmin':-1,'dfr_vmax':5})

c.plotter_cross(along_track_index=144)


 