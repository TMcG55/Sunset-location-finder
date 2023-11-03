import requests
import matplotlib.pyplot as plt
import numpy as np

def get_elevations(locations):
    base_url = "https://api.open-elevation.com/api/v1/lookup"
    coordinates = "|".join([f"{lat},{lng}" for lat, lng in locations])
    params = {
        "locations": coordinates
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        elevations = [result["elevation"] for result in data["results"]]
        return elevations
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    locations = []

    topLeft = [51.237546, -0.899248]
    bottomRight = [51.161850, -0.701617]
    y = topLeft[0]
    x = topLeft[1]
    xResolution = (topLeft[1]-bottomRight[1])/5
    yResolution = (topLeft[0]-bottomRight[0])/15
    j = 0
    while y>bottomRight[0]:
        while x<bottomRight[1]:
            locations.append([y,x])
            x = x - xResolution
        y = y - yResolution
        x = topLeft[1]
        j=j+1

    elevations = get_elevations(locations)    
    yLocals=[]
    xLocals=[]

    for i in range(len(locations)):
        yLocals.append(locations[i][0])
        xLocals.append(locations[i][1])
        
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_zlabel('Elevation')
    ax.set_title('Elevation Data')
    surf = ax.plot_trisurf(np.array(xLocals), np.array(yLocals), np.array(elevations), cmap='viridis')
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)


    plt.show()


if __name__ == "__main__":
    main()
