#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 10:24:52 2022

@author: ghiggi
"""

# -----------------------------------------------------------------------------.
#### Custom dimensions

# TRMM-CSH
# nlayer: 80

# 2A-GMI
# npixel and nray same stuff in GMI ?

## DPR
# nfreqHI: 3,
# nNode: 5, nbinSZP: 7,
# nfreq: 2, nNUBF: 3, method: 6,
# nsdew: 3, nearFar: 2, four: 4,
# nNP: 4, XYZ: 3

# nscan	    =	number	of	DPR	scans	per	granule,	approximately	7900
# nrayNS	=	49	rays	in	each	Ku	band	(NS)	scan
# nrayMS	=	25	rays	in	each	matched	Ku-Ka	(MS)	scan
# nbinC    	=	88	vertical	range	bins	at	250	m	intervals
# nbinEnv	=	10	range	bins	for	environmental	parameter	sampling
# nbinLow	=	9	range	bins	for	low-resolution	PSD	parameter	sampling
# nbinPhase	=	5	range	bins	indicating	phase	transitions
# nbinTrans	=	10	range	bins	describing	the	precipitation	liquid	phase	fraction  through	the	mixed	phase	layer

# nPSDhigh	=	1	parameters	for	describing	the	precipitation	particle	size  distribution	at	250	m	resolution
# nPSDlow	=	2	parameters	for	describing	the	precipitation	particle	size  distribution	at	low	vertical	resolution.
# nAB	=	2	power	law	parameters	to	describe	particle	densities
# nKuKa	=	2	indices	for	the	Ku	and	Ka	channels
# nchan	=	15	GMI	channels,	including	separate	accounting	for	the  double	side-band	channels.

# nBnPSD --> CMB precip profile

#### Renaming dimensions
# 2A-DPR
# - nwater --> diagnosis ---> algorithm, ancillary
# - method --> PIA_method
# - range: 176 bin at 125 m interval (FS)  (In version 7)

# CORRA:
# - nBnPSD in V7. In doc nbinC --> 88 bin at 250 m interval
# - No height variable

# CSH and SLH
# - nlayer: 80; discretized between 0-20 km

# -----------------------------------------------------------------------------.

# GPM-CMB
# Ku+GMI --> KuGMI  (before V6)
# Ku+Ka+GMI --> KuGMI  (before V7)

# TRMM CORRAT V08 --> IS V5 and V6 (different versioning)

# IMERG FR (RS) quite some latency. In August 2022, last available in September 2021
# IMERG FR available only from v6
# --> https://arthurhouhttps.pps.eosdis.nasa.gov/text/gpmallversions/V06/2021/07/01/imerg/

# 2A-SAPHIR available only in 2A-PRPS
# PRPS:
# - Currently available for V6
# - 2A SAPHIR and ... only V6 currently


# -----------------------------------------------------------------------------.
# Calibrated antenna temperature (Ta)
# Brightness temperature (Tb)
# Common intercalibrated brightness temperature (Tc) p

# The CSH latent heat product is generated using CORRA data
# The SLH is generated from DPR and PR

####-------------------------------------------------------------------------.
###########################
#### GES_DISC_PRODUCTS ####
###########################
# https://gpm1.gesdisc.eosdis.nasa.gov/data/
# https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L2/
# https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/
# GPM 2HCSH https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L2/GPM_2HCSH.07/
# GPM 2HSLH https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L2/GPM_2HSLH.07/
# GPM 2BCMB https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L2/GPM_2BCMB.07/

####-------------------------------------------------------------------------.
##################
#### Archives ####
##################
# https://arthurhouhttps.pps.eosdis.nasa.gov/text/gpmdata/
# https://arthurhouhttps.pps.eosdis.nasa.gov/text/gpmdata/gpmallversions

# --> helpdesk@pps-mail.nascom.nasa.gov

#### gpmallversions
# V06 - gprof folder not present
# --> https://arthurhouhttps.pps.eosdis.nasa.gov/text/gpmallversions/V05/2019/07/01/gprof/
# --> https://arthurhouhttps.pps.eosdis.nasa.gov/text/gpmallversions/V07/2019/07/01/gprof/
# V06 - only data till end of 2021
# V07
# - prps not available ... only V06
# - imerg not available ... only V06 and below
# --> https://arthurhouhttps.pps.eosdis.nasa.gov/text/gpmallversions/V06/2019/07/01/prps/
# - precipFeature directory available

#### gpmdata
# Since 2021/10/01 no imerg in gpmdata (and is V06)
# - https://arthurhouhttps.pps.eosdis.nasa.gov/text/gpmdata/2021/10/02/


## Old PMW:
# - CLIM prefix
# https://arthurhouhttps.pps.eosdis.nasa.gov/text/gpmallversions/V07/2014/07/01/gprof/

# ------------------------------------------------------------------------------
## RS
# https://arthurhouhttps.pps.eosdis.nasa.gov/text/gpmdata/2022/04/22/
# https://arthurhouhttps.pps.eosdis.nasa.gov/text/gpmdata/2022/04/22/radar
# https://arthurhouhttps.pps.eosdis.nasa.gov/text/gpmdata/2014/04/22/radar/

## NRT
# https://jsimpsonhttps.pps.eosdis.nasa.gov/text/
# https://jsimpsonhttps.pps.eosdis.nasa.gov/text/radar/   # 2A
# https://jsimpsonhttps.pps.eosdis.nasa.gov/text/combine/ # CORRA 2B

# -----------------------------------------------------------------------------.


# Radiometer-only estimates	of precipitation in	regions
#  where radar reflectivities are below	the	minimum	detectable of the DPR.

# The	clutter	zone	of	the	DPR	is	roughly	0.7	km	at	nadir	view,
# rising	to	over	2	km	at	the	swath	edges.

# Climatological	relationships	between	the
# near-surface	estimates	and	 “surface”	estimates	 (actually,	at	~0.8	 km	above	 the
# Earth’s	 surface),	 were	 established	 using	 nadir-view	 CORRA	 V07	 precipitation
# rate	 profile	estimates	 for	 ocean/land	and	 convective/stratiform	 classes

# Scale	the  near-surface precipitation	 rates	 to	 statistically	 estimate	 surface
# precipitation	 rates	 when	 the	 surface	 level	 is	 within	 the	 clutter


# Like TRMM, the GPM spacecraft is designed to collect data while flying either “forward” (i.e.,
# +X Flight axis along the velocity direction, yaw = 0) or “backward” (i.e., -X Flight axis along the
# velocity direction, yaw = 180 degrees). The orientation is changed about every 40 days in order
# to keep the +Y side of the spacecraft in shadow, and the –Y side in sunlight as the orbit
# precesses. This is done for the purpose of thermal control. For yaw turns, the slew periods are
# expected to be about 10 minutes.

# Delta-V Maneuvers
# The GPM spacecraft will adjust its orbit about once a week to once a month, depending on solar
# activity, which affects atmospheric drag. There is a special sequence of mode changes that will
# occur with each orbit adjust.