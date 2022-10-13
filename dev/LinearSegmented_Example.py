#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 17:09:00 2022

@author: ghiggi
"""
# In the “fig.savefig” function, setting “bbox_inches = ‘tight’” ensures the entire plot is saved, 
including the title. 

# The “dpi” parameter sets the resolution of the image in dots per inch
# Suggested guidelines for setting the dpi parameter, depending on the application of your graphic:
# 100 (default): relatively low resolution, suitable for webpages
# 150: moderate resolution, suitable for presentations
# 300: high resolution, suitable for written reports
# 600: very high resolution, suitable for scientific journal articles


 #Create custom continuous colormap for AOD data
#.set_over sets color for plotting data > max
# - https://www.star.nesdis.noaa.gov/smcd/spb/aq/STAR_Python_Hub/aod.php
color_map = mpl.colors.LinearSegmentedColormap.from_list('custom_AOD', [(0, 'indigo'),(0.1, 'mediumblue'), 
(0.2, 'blue'), (0.3, 'royalblue'), (0.4, 'skyblue'), (0.5, 'cyan'), (0.6, 'yellow'), (0.7, 'orange'), 
(0.8, 'darkorange'), (0.9, 'red'), (1, 'firebrick')], N = 150)
color_map.set_over('darkred')

#Set range for plotting AOD data (data min, data max, contour interval) (MODIFY contour interval)
#interval: 0.1 = runs faster/coarser resolution, 0.01 = runs slower/higher resolution
data_range = np.arange(0, 1.1, 0.05)

color_map.set_over('darkred')
norm = mpl.colors.Normalize(vmin = 0, vmax = 1)
cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap = color_map, norm = norm, orientation = 'horizontal', 
ticks = [0, 0.25, 0.5, 0.75, 1], extend = 'max')
cb.set_label(label = 'AOD', size = 'medium', weight = 'bold')
cb.ax.set_xticklabels(['0', '0.25', '0.50', '0.75', '1.0'])
cb.ax.tick_params(labelsize = 'medium')