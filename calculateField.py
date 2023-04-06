#%%

## precisa instalar o basemap "pip install basemap" ou "conda install basemap"
## tambem precisa do pyIGRF "pip install pyIGRF"

# from mpl_toolkits.basemap import Basemap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import matplotlib.patheffects as pe
import pyIGRF
import itertools


# %%
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def add_zebra_frame(ax, lw=2, crs="pcarree", zorder=None):

    ax.spines["geo"].set_visible(False)
    left, right, bot, top = ax.get_extent()
    
    # Alternate black and white line segments
    bws = itertools.cycle(["k", "white"])

    xticks = sorted([left, *ax.get_xticks(), right])
    xticks = np.unique(np.array(xticks))
    yticks = sorted([bot, *ax.get_yticks(), top])
    yticks = np.unique(np.array(yticks))
    for ticks, which in zip([xticks, yticks], ["lon", "lat"]):
        for idx, (start, end) in enumerate(zip(ticks, ticks[1:])):
            bw = next(bws)
            if which == "lon":
                xs = [[start, end], [start, end]]
                ys = [[bot, bot], [top, top]]
            else:
                xs = [[left, left], [right, right]]
                ys = [[start, end], [start, end]]

            # For first and lastlines, used the "projecting" effect
            capstyle = "butt" if idx not in (0, len(ticks) - 2) else "projecting"
            for (xx, yy) in zip(xs, ys):
                ax.plot(
                    xx,
                    yy,
                    color=bw,
                    linewidth=lw,
                    clip_on=False,
                    transform=crs,
                    zorder=zorder,
                    solid_capstyle=capstyle,
                    # Add a black border to accentuate white segments
                    path_effects=[
                        pe.Stroke(linewidth=lw + 1, foreground="black"),
                        pe.Normal(),
                    ],
                )



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
# %%

xspace = np.arange(-180,181,0.5)
yspace = np.arange(-90,91, 0.5)
incl, euator, magnt = calculateMag(xspace, yspace, 2022., 100)
#%%
euator = smooth(euator,10)
# %%

stations = pd.read_csv("amandastations.csv", sep=";")

circ_size=500



#######################################################################################


crs = ccrs.PlateCarree()
fig = plt.figure(figsize=(14,14))

ax = plt.axes(projection=crs)
# [-130,-20,-85,80]
# 

# ax.set_xticks((-130, -110, -90, -80))





ax.coastlines()
ax.coastlines(resolution='110m',color='k')
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.BORDERS)
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                 linewidth=1,color='gray', alpha=1, linestyle=':')
gl.xlabels_top = True
gl.ylabels_right = True
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

x = stations['long']
y = stations['lat']

for i, txt in enumerate(stations['nome']):
    if txt !="BRW":
        if txt == "AIA":
            ax.annotate(txt, (x[i]-6, y[i]), weight='bold',  
                        fontsize=12, color='b')
        elif txt == "RES":
            ax.annotate(txt, (x[i], y[i]+4), weight='bold',  
                        fontsize=12, color='b')
        else:
            ax.annotate(txt, (x[i]+3, y[i]), weight='bold',  
                        fontsize=12, color='b')
ax.scatter(x,y,s=circ_size,color='r', alpha=1)

ax.scatter(stations["long"].values, stations["lat"].values, c="red", label="High Latidues")


levels = np.arange(22000,24000,2000)

CS = ax.contour(xspace, yspace,magnt, levels=levels, cmap='jet')


ax.clabel(CS, inline=True, fontsize=10)#, manual=manual_locations)
ax.plot(xspace,euator, color="magenta",label="Magnetic equator")
# ax.set_xticks(ax.get_xticklabels(minor=False, which=None))

ax.set_extent((-140, -20, -90,90))

add_zebra_frame(ax, crs=crs)

ax.set_xticks((-130,-20))
ax.set_yticks((-85,80))

plt.ylabel("Latitude", fontsize=15, labelpad=35)
plt.xlabel("Longitude", fontsize=15, labelpad=20)
# plt.xlim(-130,-20)
# plt.ylim(-85,80)


# plt.xlabel("Longitude (Deg)")
# plt.ylabel("Latitude (Deg)")\
plt.savefig("INtensity.png", bbox_inches='tight')
# %%

# %%
