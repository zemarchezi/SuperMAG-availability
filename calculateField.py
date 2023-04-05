#%%

## precisa instalar o basemap "pip install basemap" ou "conda install basemap"
## tambem precisa do pyIGRF "pip install pyIGRF"

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.image as mpimg
import pyIGRF

# %%
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
# pr = cProfile.Profile()
# pr.enable()
xspace = np.arange(-180,181,0.5)
yspace = np.arange(-90,91, 0.5)
incl, euator, magnt = calculateMag(xspace, yspace, 2022., 100)
# pr.disable()
# %%

fig, ax = plt.subplots(figsize=(14,14))
# img=mpimg.imread('world_map.jpg')
# ax.imshow(img, extent=[-180,180,-90,90])


m = Basemap()
m.drawcoastlines(linewidth=1.0, linestyle='solid', color='black')
# m.fillcontinents()
m.drawmeridians(range(0, 360, 30), color='k', linewidth=1.0, dashes=[4, 4], labels=[0, 0, 1, 1])
m.drawparallels(range(-90, 90, 30), color='k', linewidth=1.0, dashes=[4, 4], labels=[1, 1, 0, 0])
# m.drawcountries()




# print(w_lon)
# if w_lon >= 180:
#     w_lon -= 360
#     e_lon -= 360

# for i in range(len(lons)):
#     if lons[i] >= 180:
#         lons[i] -= 360
# ax.legend(bbox_to_anchor=(1.06, 1.05))# ax.set_xlim(-180,180)
# ax.set_ylim(-90, 90)
levels = np.arange(22000,30000,2000)
# levels = np.arange(22000, 30000, 40000)
CS = ax.contour(xspace, yspace,magnt, levels=levels, cmap='jet')
# manual_locations = [
#     (21000, 24000)]
ax.clabel(CS, inline=True, fontsize=10)#, manual=manual_locations)
# ax.scatter(all_stations["GLongg"].values, all_stations["GLAT"].values, c="red")
ax.plot(xspace,euator, color="magenta",label="Magnetic equator")



plt.ylabel("Latitude", fontsize=15, labelpad=35)
plt.xlabel("Longitude", fontsize=15, labelpad=20)

# plt.xlabel("Longitude (Deg)")
# plt.ylabel("Latitude (Deg)")\
plt.savefig("INtensity.png", bbox_inches='tight')
# %%
