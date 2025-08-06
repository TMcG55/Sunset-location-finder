import rasterio
from rasterio.transform import rowcol
from pyproj import Transformer
from pyproj import Geod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
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
    # takes a sample of the elevations along the azimuth from the observer up to a given distance
    geod = Geod(ellps="WGS84")
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
    if h<0:
        h=0
    r = 6371000    #radius of earth
    d = 0 #distance from observer to point
    a = np.sqrt(((h+r)**2)-r**2)
    
    altitude_adjusted = altitude
    for i in range(len(altitude)):
        d = distance_meters[i]
        if d>a:
            x =np.sqrt((d**2)-2*d*a+(a**2)+(r**2))-r
            altitude_adjusted[i]=altitude[i]-x
    return [altitude_adjusted,distance_meters,current_altitude]

def terrain_assess_poor(elevation,distance_meters,sunset_start_altitude = 0.5,current_altitude=0):
    #attempts to roughly guage a spots quality based on 3 sun altitudes
    sunset_start_viewline_high=[]
    sunset_start_viewline_med=[]
    sunset_start_viewline_low=[]

    for i in distance_meters:
        sunset_start_viewline_high.append(current_altitude+i*np.tan(sunset_start_altitude*np.pi/180))
        sunset_start_viewline_med.append(current_altitude+i*np.tan((sunset_start_altitude/2)*np.pi/180))
        sunset_start_viewline_low.append(current_altitude+i*np.tan((1)*np.pi/180))

    # plt.plot(distance_meters,sunset_start_viewline_high)
    # plt.plot(distance_meters,sunset_start_viewline_med)
    # plt.plot(distance_meters,sunset_start_viewline_low)
    # plt.plot(distance_meters,elevation)
    # plt.show()

    i=0
    for j in elevation:
        if sunset_start_viewline_low[i]<j:
            quality=3
        elif sunset_start_viewline_med[i]<j:
            quality=2
        elif sunset_start_viewline_high[i]<j:
            quality=1
        else:
            quality=0
    return quality

def terrain_assess_fine(elevation,distance_meters,current_altitude=0,delta = 0.1):
    # uses bisectional analysis to give a better rating of sunset quality
    show_plot=False

    delta=0.05
    current_altitude=elevation[0]+0.1
    last_sun_high=10
    last_sun_low=-10
    last_sun_mid=(last_sun_high-last_sun_low)/2+last_sun_low
    i=0
    r = last_sun_high-last_sun_low
    high_flag=True
    low_flag=True
    mid_flag=True
    overbounds=False
    underbounds=False
    top_half=False
    bottom_half=False
    steps=0
    if show_plot:
        plt.plot(elevation)
    while r>delta:
        if high_flag:
            last_sun_high_line=[]
            for j in distance_meters:
                last_sun_high_line.append(current_altitude+j*np.tan(last_sun_high*np.pi/180))
        if low_flag:
            last_sun_low_line=[]
            for j in distance_meters:
                last_sun_low_line.append(current_altitude+j*np.tan((last_sun_low)*np.pi/180))
        if mid_flag:
            last_sun_mid_line=[]
            for j in distance_meters:
                last_sun_mid_line.append(current_altitude+j*np.tan((last_sun_mid)*np.pi/180))
        high_flag=True
        low_flag=True
        mid_flag=True
        mid_check_only_flag=False


        overbounds=False
        underbounds=False
        top_half=False
        bottom_half=False
        for k in range(len(distance_meters)-1):
            k=k+1
            if not mid_check_only_flag:
                if elevation[k]>last_sun_high_line[k]:
                    overbounds=True
                elif elevation[k]>last_sun_mid_line[k]:
                    top_half=True
                elif elevation[k]>last_sun_low_line[k]:
                    bottom_half=True
                else:
                    underbounds=True
            else:
                if elevation[k]>last_sun_mid_line[k]:
                    top_half=True
                else:
                    bottom_half=True
            mid_check_only_flag=True

        if overbounds:
            rating=0
            plt.clf()
            return rating
        elif top_half:
            last_sun_low=last_sun_mid
            last_sun_low_line=last_sun_mid_line
            last_sun_mid=(last_sun_high-last_sun_low)/2+last_sun_low
            r=last_sun_high-last_sun_low
            high_flag=False
            low_flag=False
        elif bottom_half:
            last_sun_high=last_sun_mid
            last_sun_high_line=last_sun_mid_line
            last_sun_mid=(last_sun_high-last_sun_low)/2+last_sun_low
            r=last_sun_high-last_sun_low
            high_flag=False
            low_flag=False
        elif underbounds:
            last_sun_high_line=last_sun_low_line
            last_sun_high=last_sun_low
            high_flag=False
            last_sun_low=0
        if show_plot:
            #plt.plot(last_sun_high_line,color = 'red')
            plt.plot(last_sun_mid_line,color = 'k')
            #plt.plot(last_sun_low_line,color = 'blue')
        steps=steps+1
        if steps>10:
            print('timed out. high:',last_sun_high,' low: ',last_sun_low,' mid: ', last_sun_mid)
            rating=10*np.exp(-0.2*last_sun_mid)
            if show_plot:
                plt.show()
                print(rating)
            return rating

    rating=10*np.exp(-0.2*last_sun_mid)
    if show_plot:
        plt.show()
        print(rating)

    
    return rating

def solar_postion_time_calcs(current_location):
    #calculates the solar position and time of sunset for a given location
    #local time
    local_timezone = pytz.timezone('Canada/Pacific')
    date_local = local_timezone.localize(pd.Timestamp.now())
    date_utc=date_local.astimezone(pytz.utc)
    #calculate sunrise and sunset time
    site=Location(latitude=current_location[1], longitude=current_location[0], tz="Canada/Pacific")
    times=pd.date_range(date_local, periods=288, freq='5min', tz=site.tz)  #sample over 24h every 5min
    rise_set_transit=site.get_sun_rise_set_transit(times)
    sset=rise_set_transit['sunset'].iloc[0]
    srise=rise_set_transit['sunrise'].iloc[0]
    #print('sunset time: ',sset)
    labels = ['sunset','sunrise']
    i=0
    azimuths = [0,0]

    while i < 2: 
        solpos = pvlib.solarposition.get_solarposition(rise_set_transit[labels[i]].iloc[0],current_location[1],current_location[0], method='nrel_numpy') # takes lat then long 
        azimuths[i] = solpos["azimuth"].iloc[0]
        i=i+1

    geod = Geod(ellps="WGS84")
    sunset_az_terminus = geod.fwd(current_location[0],current_location[1],azimuths[0],10000,)
    sunrise_az_terminus = geod.fwd(current_location[0],current_location[1],azimuths[1],10000,)
    return azimuths,sunset_az_terminus,sunrise_az_terminus

#data_convert()

current_location = [-125.906616,49.152985,] # long lat
#current_location = [-125.4,49.150,] # long lat

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


delta = 0.01
length = 5
long_min=current_location[0]-delta*length
long_max=current_location[0]+delta*length
lat_min=current_location[1]-delta*length
lat_max=current_location[1]+delta*length
current_long=long_min
current_lat=lat_min
sunset_start_altitude=10
rating = []
counter=0

while current_long<=long_max:
    print('new long:', counter)
    while current_lat<=lat_max:
        azimuths,sunset_az_terminus,sunrise_az_terminus = solar_postion_time_calcs([current_long,current_lat])
        terrain_adjusted = terrain_linesample([current_long,current_lat],azimuths[0],10000,elevation)
        terrain_assess_result=terrain_assess_fine(terrain_adjusted[0],terrain_adjusted[1],delta=0.1)
        rating.append([terrain_assess_result,current_long,current_lat])
        current_lat=current_lat+delta
    current_lat=lat_min
    current_long=current_long+delta
    counter=counter+1

rating=np.array(rating)

# azimuths,sunset_az_terminus,sunrise_az_terminus = solar_postion_time_calcs(current_location)

# terrain_adjusted = terrain_linesample(current_location,azimuths[0],10000,elevation)

# sunset_start_altitude=10

# terrain_aseess_result=terrain_assess_poor(terrain_adjusted[0],terrain_adjusted[1],sunset_start_altitude,terrain_adjusted[2])
# good_rating=terrain_assess_fine(terrain_adjusted[0],terrain_adjusted[1],sunset_start_altitude,terrain_adjusted[2])


#colours = ['red','orange','blue','green']
#cmap = ListedColormap(colours)
#norm = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=cmap.N)
#plt.imshow(rating[1],rating[2],rating[0])
#plt.show()


plt.imshow(elevation, cmap='terrain', extent=extent)
plt.scatter(rating[:,1], rating[:,2],c=rating[:,0],marker='x',cmap='hot')
#plt.plot([current_location[0],sunset_az_terminus[0]],[current_location[1],sunset_az_terminus[1]], color='darkorange', linestyle='-', linewidth=2)#sunset
#plt.plot([current_location[0],sunrise_az_terminus[0]],[current_location[1],sunrise_az_terminus[1]], color='deepskyblue', linestyle='-', linewidth=2)#sunrise

#plt.plot(current_location[0], current_location[1], marker='x', color='red', markersize=10, label="Target Point")
# plt.plot([current_location[0],current_location[0]+10000],[current_location[1],current_location[1]+10000], color='k', linestyle='--', linewidth=1)
# plt.plot([current_location[0],current_location[0]-10000],[current_location[1],current_location[1]+10000], color='k', linestyle='--', linewidth=1)
# plt.plot([current_location[0],current_location[0]-10000],[current_location[1],current_location[1]-10000], color='k', linestyle='--', linewidth=1)
# plt.plot([current_location[0],current_location[0]+10000],[current_location[1],current_location[1]-10000], color='k', linestyle='--', linewidth=1)

#plt.colorbar(label='Elevation (m)')
plt.axis('equal')
plt.show()

