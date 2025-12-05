"""
Project 3 functions module
"""

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

class RandomWalker:
    """
    Random walker class
    """

    def __init__(self, n_walkers=5, n_steps=10, bounds_min=[0,0,0], bounds_max = [10,10,10],
                 step_size=1.0, walker_radius=1.0, atom_cords=[[0,0,0]], atom_radii=[0]):

        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.bounds_min = np.array(bounds_min)
        self.bounds_max = np.array(bounds_max)
        self.step_size = step_size
        self.walker_radius = walker_radius
        self.atom_cords = np.array(atom_cords)
        self.atom_radii = np.array(atom_radii)

    def generate_walkers(self):
        """
        generates walkers slowly,
        returns list of starting positions 
        uniformly distributed in the given bounds
        """
        walkers = []
        for _ in range(self.n_walkers):
            pos = np.random.uniform(self.bounds_min, self.bounds_max)
            walkers.append(pos)
        return walkers

    def generate_walkers_fast(self):
        """
        Generates walkers fast(er),
        vectorized version that generates an array with shape (n_walkers, 3)
        of uniformly distributed starting coordinates
        """
        return np.random.uniform(self.bounds_min, self.bounds_max,(self.n_walkers, 3))

    def walk(self, walkers):
        """
        Slower walking method, iterates over starting positions of walkers,
        then walks for each walker in random directions. 
        Checks for collisions with objects in the space,
        and enforces the boundaries of the space.
        returns list of np arrays containing path taken by walker
        """
        paths = []
        for pos in walkers:
            path = [pos.copy()]
            for _ in range(self.n_steps):
                direction = np.random.randint(-1,2,3)
                new_pos = pos + direction * self.step_size

                if np.any(new_pos < self.bounds_min) or np.any(new_pos > self.bounds_max):
                    continue
                if collision(new_pos, self.atom_cords, self.atom_radii, self.walker_radius):
                    continue

                pos = new_pos
                path.append(pos.copy())

            paths.append(np.array(path))

        return paths

    def walk_fast(self, walkers):
        """
        Faster walking method that uses more or less the same method as the slower walker,
        except we generate an array of (n_walkers, n_steps, 3) shape holds directional
        choices for each walker before the loop instead of doing it inside the for loop,
        cutting down the amount of randint calls we do drastically from n_steps*n_walkers to 1.
        """
        paths = []
        n = self.n_walkers
        steps = self.n_steps
        step_size = self.step_size

        directions = np.random.randint(-1,2,(n, steps, 3))*step_size


        for i in range(n):
            pos = walkers[i].copy()
            path = [pos.copy()]

            for step in directions[i]:
                new_pos = pos + step

                if np.any(new_pos < self.bounds_min) or np.any(new_pos > self.bounds_max):
                    continue

                if collision(new_pos, self.atom_cords, self.atom_radii, self.walker_radius):
                    continue

                pos = new_pos
                path.append(pos.copy())
            paths.append(np.array(path))

        return paths



def estimate_volume(n_walkers=100, n_steps=500,
                    bounds_min=[0,0,0], bounds_max = [20,20,20],
                    step_size=1.0, walker_radius=1.0,
                    atom_cords=[[0,0,0]], atom_radii=[0], use_dna_data = True):
    """
    Main function that handles the logic and function calls. Calls helper functions for loading dna data and calculating bounds for dna if we use_dna_data = True.
    Makes calls to create RandomWalker class object, generates walkers(fast),
    runs the walks, creates cell grid.
    It also resamples starting position of a walker if it starts inside an object
    and finally calculates the total amount of cells, cells visited and the fraction of cells visited.
    Returns paths, lower bounds, upper bounds, fraction of cells visited, 
    total amount of cells, amount of cells visited and the grid map itself.
    """

    assert n_walkers > 0 and n_steps > 0, "Need more than 0 walkers and steps"

    if use_dna_data:
        atom_cords, atom_radii = load_dna_data()
        bounds_min, bounds_max = calc_bounds(atom_cords, atom_radii)
    else:
        atom_cords = np.array(atom_cords, dtype=float)
        atom_radii = np.array(atom_radii, dtype=float)
        bounds_min = np.array(bounds_min, dtype=float)
        bounds_max = np.array(bounds_max, dtype=float)


    rw = RandomWalker(n_walkers, n_steps, bounds_min, bounds_max, step_size, walker_radius, atom_cords, atom_radii)
    walkers = rw.generate_walkers_fast()

    for i in range(len(walkers)):
        while collision(walkers[i], atom_cords, atom_radii, walker_radius):
            walkers[i] = np.random.uniform(rw.bounds_min, rw.bounds_max)

    paths = rw.walk(walkers)

    grid_map = create_grid_map(paths,bounds_min, bounds_max,1)
    t_cells = grid_map.size
    v_cells = np.sum(grid_map)
    frac = v_cells/t_cells

    return paths, bounds_min, bounds_max, frac, v_cells, t_cells, grid_map

def collision(pos, atom_cords, atom_radii, walker_radius):
    """
    Creates an array of vectors from the walkers position to all atom coordinates
    and checks if any of the distances are
    smaller than the sum of any atoms radius and the walker radius.
    Returns true if it is the case(indicating a collision)
    """

    if atom_cords is None or len(atom_cords) == 0:
        return False

    distance = np.sqrt(np.sum((atom_cords - pos)**2,axis=1))
    return np.any((distance) < (atom_radii + walker_radius))


def create_grid_map(paths, bounds_min, bounds_max, cell_size=1):
    """
    Creates a 3D bit map based on the size of the given space and cell sizes to represent visisted and unvisited cells.
    Iterates over each path of the walkers and converts the positions into cell coordinates.
    Any cell that a walker has pathed through is considered clear since a collision would skip the coordinate.
    Checks if any path exits bounds, if not marks cell as visited. 
    Returns numpy array of visited(1) and unvisited(0) cells
    """
    size = bounds_max - bounds_min
    grid_shape = np.array(size / cell_size, dtype=int)
    grid_map = np.zeros(grid_shape)


    for path in paths:
        for pos in path:
            i = int((pos[0] - bounds_min[0]) / cell_size)
            j = int((pos[1] - bounds_min[1]) / cell_size)
            k = int((pos[2] - bounds_min[2]) / cell_size)

            if 0 <= i < grid_shape[0] and 0 <= j < grid_shape[1] and 0 <= k < grid_shape[2]:
                grid_map[i, j, k] = 1

    return grid_map


def load_dna_data(filename="dna_coords.txt"):
    """
    Loads DNA data from the given file(dna_coords.txt by default)
    and pairs the given atom with the atomic radius of the corresponding atom.
    Returns two numpy arrays with corresponding coordinates and radii in angstroms(1e^-10m)
    """
    atom_radii = {
        "H": 1.2,
        "C": 1.7,
        "N": 1.55,
        "O": 1.52,
        "P": 1.8,
    }

    coords = []
    radii = []

    with open(filename, "r") as f:
        for line in f:
            atom = line.split()
            coords.append([atom[1],atom[2],atom[3]])
            radii.append(atom_radii[atom[0]])

    f.close()

    dna_coords = np.array(coords, dtype=float)
    dna_radii = np.array(radii, dtype=float)

    return dna_coords, dna_radii

def calc_bounds(atom_cords, atom_radii, walker_radius=1.0, margin=2.0):
    """
    Calculates bounds_min and bounds_max based on the largest and
    smalles value of x,y, and z of an atom(including its radius), a margin and the walker radius.
    Returns numpy arrays of lower bounds and upper bounds
    """

    min_xyz = np.min(atom_cords - atom_radii[:, np.newaxis], axis=0)
    max_xyz = np.max(atom_cords + atom_radii[:, np.newaxis], axis=0)
    bounds_min = min_xyz - walker_radius - margin
    bounds_max = max_xyz + walker_radius + margin

    return bounds_min, bounds_max

def estimate_empty_space_test():
    """
    Estimates volume of an empty space as the simplest form of testing functionality, accuracy and parameters.
    Returns plot of walker paths in 3D space
    """
    paths, bounds_min, bounds_max, frac, v, t, _ = estimate_volume(
                n_walkers=100,
                n_steps=800,
                bounds_min=[0,0,0],
                bounds_max = [20,20,20],
                step_size=1.0,
                walker_radius=1.0,
                use_dna_data=False)


    print("Total amount of cells/volume in Angstroms: ", t)
    print("Amount of visited cells/volume in Angstroms: ", v)
    print("Fraction of cells visited: ", frac)

    return plot_walks(paths, bounds_min, bounds_max)

def estimate_sphere_test():
    """
    Estimates volume of a space with a single sphere in the middle for predictible results to measure accuracy based on parameters.
    Returns a 2D projection for each axis.
    """
    _, _, _, frac, v, t, grid_map = estimate_volume(n_walkers=100,
                    n_steps=800,
                    bounds_min=[0,0,0],
                    bounds_max = [20,20,20],
                    step_size=1.0,
                    walker_radius=1.0,
                    atom_cords=[[10.0,10.0,10.0]],
                    atom_radii=[5.0],
                    use_dna_data=False)

    print("Total amount of cells/volume in Angstroms: ", t)
    print("Amount of visited cells/volume in Angstroms: ", v)
    print("Fraction of cells visited: ", frac)
    return plot_2d_projection(grid_map)

def boundary_test():
    """
    Test function for checking if a walker ever exits bounds
    """

    rw = RandomWalker(n_walkers=10, n_steps = 10, bounds_min=[0,0,0], bounds_max=[5,5,5])
    walkers = rw.generate_walkers_fast()
    paths = rw.walk(walkers)

    for path in paths:
        assert np.all(path >= rw.bounds_min) and np.all(path <= rw.bounds_max), "Walker out of bounds"

    return print("Walkers stayed within bounds")

def plot_walks(paths, bounds_min, bounds_max):
    """
    3D plot of each walker path
    """
    ax = plt.figure().add_subplot(projection='3d')

    for path in paths:
        ax.plot(path[:,0],path[:,1],path[:,2])

    ax.set_xlim(bounds_min[0], bounds_max[0])
    ax.set_ylim(bounds_min[1], bounds_max[1])
    ax.set_zlim(bounds_min[2], bounds_max[2])
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    plt.show()

def plot_2d_projection(cells):
    """
    Plotting function shows 2D projections of the spacial grid map along each axis.
    Colours indicate the sum of visisted cells along said axis.
    """
    axis = ["X", "Y", "Z"]

    for i in range(3):
        plt.figure()
        projection = np.sum(cells, axis=i)
        plt.imshow(projection)
        plt.title("2D projection of visited cells along axis "+ axis[i])
        plt.show()
