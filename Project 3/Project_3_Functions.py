import matplotlib.pyplot as plt
import random as rng
import numpy as np


class Sphere: # Creating a sphere class. This makes it easier to handle the spheres later.
    def __init__(self, center, radius): # A sphere is defined by it's center point and it's radius.
        self.center = center
        self.radius = radius


def create_box(x, y, z): # This function simply creates a numpy array with the desired dimensions of the box.
    
    return np.array([float(x), float(y), float(z)])


def create_points(N, box): # This function creates N random points within the volume of the box. I let it be able to create several points, as this is usefull later.
    
    points = []

    for i in range(N):
        x = rng.uniform(0, box[0])
        y = rng.uniform(0, box[1])
        z = rng.uniform(0, box[2])

        points.append(np.array([x, y, z]))

    return np.array(points) # The output is a numpy array of numpy arrays.


def create_sphere(box): #This functions generates a random point inside a box, looks at the closest wall and selects a radius between 0 and the distance to that wall.

    center = create_points(1, box)
    distances_from_walls = [center[0, 0], box[0] - center[0, 0], center[0, 1], box[1] - center[0, 1], center[0, 2], box[2] - center[0, 2]]
    radius = rng.uniform(0, min(distances_from_walls))

    return Sphere(center, radius) # Returns a sphere object.


def inside_sphere(point, center, radius): # This function checks wether a point is inside the sphere, based on the point coordinates and the sphere dimensions.

    if (point[0] - center[0, 0])**2 + (point[1] - center[0, 1])**2 + (point[2] - center[0, 2])**2 <= radius**2:
        return True
    else:
        return False # The function returns a True or False boolean.
    

def sort_points(points, center, radius): # A function to sort the inside and outside points in two different lists.
    
    inside_points = []
    outside_points = []

    for point in points: # Checking each point and adding to corresponding list.
        if inside_sphere(point, center, radius):
            inside_points.append(point)
        else:
            outside_points.append(point)

    ratio = len(inside_points)/(len(inside_points) + len(outside_points)) # The ratio of inside points to total points.

    return np.array(inside_points), np.array(outside_points), ratio # Returns the two lists


def scatter_plot(inside_points, outside_points): # A function that handles all the plotting.
    ax = plt.figure().add_subplot(projection='3d')
    
    if len(inside_points) > 0: 
        ax.scatter(inside_points[:,0], inside_points[:,1], inside_points[:,2], color='orange', label='Inside')
    #if len(outside_points) > 0: 
    #    ax.scatter(outside_points[:,0], outside_points[:,1], outside_points[:,2], color='blue', label='Outside')
    # The blue outside points plotting has been turned of to better visualize the sphere of inside points.

    plt.legend()
    plt.show()


def find_ratio(points, spheres): # Finding the ratio of inside points to total points, when you have several spheres in the box.
    
    # Defining lists and initials:
    ratio = []
    inside_points = []
    outside_points = []
    number_of_points_checked = 0
    number_of_points_inside = 0

    for point in points: # Going through each point.
        inside = False # Defaulting the boolean to False before checking the next point.

        for sphere in spheres: # Going through each sphere.
            
            if inside_sphere(point, sphere.center, sphere.radius):
                inside = True
                break # If the point is inside a sphere, the loop ends and everything is tracked.

        if inside:
            number_of_points_inside += 1
            inside_points.append(point)

        else:
            outside_points.append(point)
        
        number_of_points_checked += 1 # Keeping track of how many points has been checked.
        
        ratio.append(number_of_points_inside/number_of_points_checked) # 'ratio' is a list of all the ratio values over time.
    
    x_values = np.linspace(0, number_of_points_checked, number_of_points_checked)

    return ratio, x_values, np.array(inside_points), np.array(outside_points)


def find_radius(element): # To find the radius based on the element symbol:
    if element == 'H':
        radius = 120e-12
    elif element == 'O':
        radius = 152e-12
    elif element == 'N':
        radius = 155e-12
    elif element == 'C':
        radius = 170e-12
    elif element == 'P':
        radius = 180e-12
    
    return radius