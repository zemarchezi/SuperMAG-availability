# -*- coding: utf-8 -*-
#==============================================================================
#--- @author: Honda, R. H., on: Wed Nov  9 13:36:31 2022
#==============================================================================
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import pyIGRF
#%%

arg_stations = pd.read_csv('amandastations.csv',delimiter=";")
def calculateMag(xspace, yspace, year, height):
    euator = np.zeros((len(xspace), 2))
    inclination = np.zeros((len(yspace), len(xspace)))
    magnt = np.zeros((len(yspace), len(xspace)))
    for x in range(len(xspace)):
        for y in range(len(yspace)):
            decl, inc, hMag, xMag, yMag, zMag, fMAg = pyIGRF.igrf_value(yspace[y], xspace[x], height, year)
            inclination[y,x] = inc
            magnt[y,x] = fMAg



    
    equator = []
    for ii in range(inclination.shape[1]):
        
        temp = inclination[:,ii]

        sts = np.where((temp > -1) & (temp < 1))
        # print(sts)

        idx = sts[0][np.argmin(abs(temp[sts]))]


        equator.append(yspace[idx])    

    return inclination, equator, magnt

#%%
xspace = np.arange(-180,181,0.5)
yspace = np.arange(-90,91, 0.5)
incl, euator, magnt = calculateMag(xspace, yspace, 2022., 100)
#%%
# N = 50
x = arg_stations['long']
y = arg_stations['lat']
# colors = np.random.rand(N)
# area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# plt.scatter(x, y, alpha=0.5)


# plt.show()



plt.rcParams.update({'font.size': 25})

fig = plt.figure(figsize=(20,20))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines(resolution='110m',color='k')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)


gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=1,color='gray', alpha=1, linestyle=':')
gl.xlabels_top = False
gl.ylabels_right = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

extend = [-130,-20,-85,80]
ax.set_extent(extend)

circ_size=500
#-----------------------------
#x = chi_stations['Longitud']
#y = chi_stations['Latitud']

#ax.scatter(x, y,s=circ_size, alpha=0.5)

#-----------------------------
x = arg_stations['long']
y = arg_stations['lat']

for i, txt in enumerate(arg_stations['nome']):
    ax.annotate(txt, (x[i], y[i]))

ax.scatter(x,y,s=circ_size,color='r', alpha=1)

#eq.sort_values( by=["long", "lat"])[["long", "lat"]]
# eq=eq.sort_values(
#      by="long",
#      ascending=True,
#      kind="mergesort"
# )

levels = np.arange(20000,23000,2000)
# levels = np.arange(22000, 30000, 40000)
CS = ax.contour(xspace, yspace,magnt, levels=levels, cmap='jet')
ax.plot(xspace,euator, color="magenta",label="Magnetic equator")
# a = eq['long']
# b = eq['lat']


# ax.plot(a,b,color='b', alpha=1)



plt.savefig('my_plot.png')







# %%