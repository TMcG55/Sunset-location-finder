import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
from pyproj import Geod
import numpy as np
import matplotlib.pyplot as plt
import pvlib
from pvlib.location import Location
import pandas as pd
import pytz
from rasterio.warp import calculate_default_transform, reproject, Resampling

src_path = "output_hh.tif"
dst_path = "output_hh_latlon.tif"

def data_convert():
    with rasterio.open(src_path) as src:
        dst_crs = "EPSG:4326"  # Target CRS
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        with rasterio.open(dst_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.bilinear)
#print('raster open')

def terrain_linesample(current_location,azimuth,distance,elevation):
    i=0
    distances=[]
    distance_meters = []
    #distances = np.array(distances)
    while i<=distance:
        distance_meters.append(i)
        i=i+30
        distances.append(geod.fwd(current_location[0],current_location[1],azimuth,i))
    i=0
    altitude = []
    while i<len(distances):
        row,col = rowcol(transform,distances[i][0],distances[i][1])
        altitude.append(elevation[row,col])
        i=i+1

    current_loc_indexx,current_loc_indexy = rowcol(transform,current_location[0],current_location[1])
    current_altitude=elevation[current_loc_indexx,current_loc_indexy]

    #drop due to curvature of earth
    h = current_altitude        #current altitude
    r = 6371000    #radius of earth
    d = 0 #distance from observer to point
    a = np.sqrt(((h+r)**2)-r**2)
    
    altitude_adjusted = altitude
    for i in range(len(altitude)):
        d = distance_meters[i]
        if d>a:
            x =np.sqrt((d**2)-2*d*a+(a**2)+(r**2))-r
            altitude_adjusted[i]=altitude[i]-x
            
            
    
    
    print('distance to horizon =',a)
    

   
    return altitude

#data_convert()

current_location = [-125.906616,49.152985,] # long lat

with rasterio.open("output_hh_latlon.tif") as src:
    # print("CRS:", src.crs)
    # print("Bounds:", src.bounds)
    # print("Transform:", src.transform)
    elevation = src.read(1)
    transform = src.transform


rows, cols = np.meshgrid(np.arange(elevation.shape[0]),np.arange(elevation.shape[1]), indexing='ij')
lons, lats = rasterio.transform.xy(transform, rows, cols)

#remove null coords
np.where(elevation[0]<=0,0,elevation[0])
elevation[elevation < -3000] = 0

# Plot with georeferenced extent
extent = [lons.min(), lons.max(), lats.min(), lats.max()]
height = int(lats.max() - lats.min())
width = int(lons.max() - lons.min())
print('raster converted')

#instantiate transforms
to_gps = Transformer.from_crs("EPSG:3979","EPSG:4326" ,always_xy=True)
to_epsg3979 = Transformer.from_crs("EPSG:4326","EPSG:3979" ,always_xy=True)

#local time
local_timezone = pytz.timezone('Canada/Pacific')
date_local = local_timezone.localize(pd.Timestamp.now())
date_utc=date_local.astimezone(pytz.utc)

#calculate sunrise and sunset time
site = Location(latitude=current_location[1], longitude=current_location[0], tz="Canada/Pacific")
times = pd.date_range(date_local, periods=288, freq='5min', tz=site.tz)  #sample over 24h every 5min
rise_set_transit = site.get_sun_rise_set_transit(times)
srise=rise_set_transit['sunset'].iloc[0]
sset=rise_set_transit['sunrise'].iloc[0]
print('sunrise and set time calculated:')
print('-rise at ',srise)
print('-set at ',sset)

labels = ['sunset','sunrise']
i=0
azimuths = [0,0]

while i < 2: 
    solpos = pvlib.solarposition.get_solarposition(rise_set_transit[labels[i]].iloc[0],current_location[1],current_location[0], method='nrel_numpy') # takes lat then long 
    azimuths[i] = solpos["azimuth"].iloc[0]
    print(solpos)
    i=i+1

geod = Geod(ellps="WGS84")
sunset_az_terminus = geod.fwd(current_location[0],current_location[1],azimuths[0],10000,)
sunrise_az_terminus = geod.fwd(current_location[0],current_location[1],azimuths[1],10000,)

terrain_linesample(current_location,azimuths[0],10000,elevation)

plt.imshow(elevation, cmap='terrain', extent=extent)
plt.plot([current_location[0],sunset_az_terminus[0]],[current_location[1],sunset_az_terminus[1]], color='darkorange', linestyle='-', linewidth=2)#sunset
plt.plot([current_location[0],sunrise_az_terminus[0]],[current_location[1],sunrise_az_terminus[1]], color='deepskyblue', linestyle='-', linewidth=2)#sunrise

#plt.plot(current_location[0], current_location[1], marker='x', color='red', markersize=10, label="Target Point")
# plt.plot([current_location[0],current_location[0]+10000],[current_location[1],current_location[1]+10000], color='k', linestyle='--', linewidth=1)
# plt.plot([current_location[0],current_location[0]-10000],[current_location[1],current_location[1]+10000], color='k', linestyle='--', linewidth=1)
# plt.plot([current_location[0],current_location[0]-10000],[current_location[1],current_location[1]-10000], color='k', linestyle='--', linewidth=1)
# plt.plot([current_location[0],current_location[0]+10000],[current_location[1],current_location[1]-10000], color='k', linestyle='--', linewidth=1)

#plt.colorbar(label='Elevation (m)')
plt.axis('equal')
plt.show()
