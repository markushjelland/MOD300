from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from astropy import units as u
from mw_plot import MWFaceOn
from mw_plot import MWSkyMap
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from ebola_data import t_g, O_g, t_l, O_l, t_s, O_s, ebola_model_plot

def plot_faceon(x, y, r, title):

    """
    Ploting a face-on view of the milky way with a point overlaid.

    The x and y inputs are the coordinates of the overlaid point in galactocentric units (kpc). The code is heavily
    based on the code provided through 'PyGaLaXy.ipynb'.


    """

    mw = MWFaceOn(
        radius = r * u.kpc,
        unit = u.kpc,
        coord = "galactocentric",
        annotation = True,
        figsize = (10, 8))

    mw.title = str(title)

    mw.scatter(x * u.kpc, y * u.kpc, c="r", s=2)


def plot_skymap(c, r, title, save = False):
    
    """
    Ploting a map of the sky centered at a given position.

    This function takes the center of the constellation and the radius we want to see around it, essentially the
    zoom. The function returns the figure made by matplotlib. This function also uses the 'Mellinger color optical
    survey' as a background color.

    The function also saves the image is requested.
    """

    mw = MWSkyMap(
        center = str(c),
        radius = (r, r) * u.arcsec,
        background = "Mellinger color optical survey")

    mw.title = str(title)
    fig, ax = plt.subplots(figsize=(10, 8))

    mw.transform(ax)

    if save:
        mw.savefig('galaxy.png')

    fig.canvas.draw()

    return fig


def plot_rgb_array(fig):
    
    """
    A function that transforms a matplotlib figure to a 3d rgb numpy array 

    The function takes the matplotlib figure as an input and returns the numpy array containing all the rgb values.
    The figure is first adjusted to match the visible canvas.
    """

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.canvas.draw()

    rgba_buf = fig.canvas.buffer_rgba()
    w, h = fig.canvas.get_width_height()

    rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))
    return rgba_arr[:, :, :3]


def encode_grayscale(rgb_array):
    
    """
    This function is an adaptation of the one found in 'PyGaLaXy.ipynb'. It is used to find the grayscale or the
    brightness of the pixels.

    It takes the previously created rgb array as an input and outputs the pixels that get a higher value as a
    grayscaled pixel, than 230. My version of this function uses the ij and ig the opposite way around, as this ended
    up beeing the correct orientation relative to the picture.
    """

    grey = np.sum(rgb_array[: , : , :] * np.array([0.299, 0.587, 0.114]), axis=2)
    x, y = [], []
    
    for ig, g in enumerate(grey):
        
        for ij, j in enumerate(g):
            
            if j > 230:
                x.append(ij)
                y.append(ig)

    return x, y


def encode_yellow(rgb_array):

    """
    This function is used to encode the yellowest pixels in the picture.

    The function takes the rgb pixel array and finds the pixels that score higher than 35 based on the
    'yellow_strenth' formula.
    """

    R = rgb_array[:, :, 0].astype(float)
    G = rgb_array[:, :, 1].astype(float)
    B = rgb_array[:, :, 2].astype(float)

    yellow_strength = (R + G)/2 - B
    mask_yellow = yellow_strength > 35
    
    y, x = np.where(mask_yellow)

    return x, y


def encode_blue(rgb_array):

    """
    This function is used to encode the bluest pixels in the picture.

    The function essentially does exactly the same thing as the 'encode_yellow' function, but for blueness. It uses
    the 'blue_strength' formula that is just the negative of the 'yellow_strength' formula (blue and yellow are
    opposites).
    """

    R = rgb_array[:, :, 0].astype(float)
    G = rgb_array[:, :, 1].astype(float)
    B = rgb_array[:, :, 2].astype(float)

    blue_strength = B - (R + G)/2
    mask_blue = blue_strength > 55

    y, x = np.where(mask_blue)

    return x, y


def scatter_plot(x_1, y_1, x_2, y_2, x_3, y_3, rgb=None, label_1='cluster 1', label_2='cluster 2', label_3='cluster 3', color_1='black', color_2='orange', color_3='blue', show=True):
    
    """
    This function is used to scatter the different pixels to highlight the bright, yellow and blue spots.

    The function takes the different x and y coordinates of the points to be ploted. It also takes the rgb variable
    created earlier, to plot the picture in task 6. The function also crops the plot to get rid of any edge colors.
    
    It first concatenates all points and optionally inserts an RGB background image behind the scatter. The
    function hides axes, maintains equal aspect ratio, and returns the resulting Matplotlib figure and axes.
    """

    x_1 = np.asarray(x_1)
    y_1 = np.asarray(y_1)
    x_2 = np.asarray(x_2)
    y_2 = np.asarray(y_2)
    x_3 = np.asarray(x_3)
    y_3 = np.asarray(y_3)

    xs = np.concatenate([x_1, x_2, x_3])
    ys = np.concatenate([y_1, y_2, y_3])
    
    cols = (
        [str(color_1)] * len(x_1)
        + [str(color_2)] * len(x_2)
        + [str(color_3)] * len(x_3))

    x_min = xs.min() + 100
    x_max = xs.max() - 100

    mask = (xs >= x_min) & (xs <= x_max)

    x1_c = x_1[(x_1 >= x_min) & (x_1 <= x_max)]
    y1_c = y_1[(x_1 >= x_min) & (x_1 <= x_max)]

    x2_c = x_2[(x_2 >= x_min) & (x_2 <= x_max)]
    y2_c = y_2[(x_2 >= x_min) & (x_2 <= x_max)]

    x3_c = x_3[(x_3 >= x_min) & (x_3 <= x_max)]
    y3_c = y_3[(x_3 >= x_min) & (x_3 <= x_max)]

    fig, ax = plt.subplots(figsize=(8, 8))

    if rgb is not None:
        H, W = rgb.shape[:2]

        ax.imshow(rgb, origin='upper', extent=(0, W, H, 0), zorder=0, alpha=0.8)

    ax.scatter(x1_c, y1_c, s=0.1, color=color_1, label=label_1, zorder=1)
    ax.scatter(x2_c, y2_c, s=0.1, color=color_2, label=label_2, zorder=1)
    ax.scatter(x3_c, y3_c, s=0.1, color=color_3, label=label_3, zorder=1)

    ax.legend(markerscale=10, fontsize=8, framealpha=0.7)

    ax.axis('off')
    ax.set_aspect('equal')

    if show:
        plt.show()

    return fig, ax


def brightness_from_coords(rgb, x, y):

    """
    This function is computing the brightness for the x and y coordinates it gets.

    It takes the rgb array and the x and y coordinates as inputs. It the returns the brightness of that pixel.

    For each (x, y) point, it retrieves the corresponding RGB triplet and computes brightness using a standard
    brightness formula. The result is a list of brightness values that can be used for clustering.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    H, W = rgb.shape[:2]

    x_int = x.astype(int)
    y_int = y.astype(int)

    rgb_vals = rgb[y_int, x_int].astype(float)

    R = rgb_vals[:, 0]
    G = rgb_vals[:, 1]
    B = rgb_vals[:, 2]

    brightness = 0.2126 * R + 0.7152 * G + 0.0722 * B
    
    return brightness


def yellowness_from_coords(rgb, x, y):
    """
    This function computes the 'yellowness' of a pixel quite like the brightness function, but for yellow.

    It uses the same 'yellowness' formula as previously used in 'encode_yellow'.
    """

    x = np.asarray(x)
    y = np.asarray(y)

    rgb_vals = rgb[y.astype(int), x.astype(int)].astype(float)

    R = rgb_vals[:, 0]
    G = rgb_vals[:, 1]
    B = rgb_vals[:, 2]

    yellowness = (R + G) / 2 - B
    
    return yellowness


def kmeans_cluster_brightness_and_plot(x_1, y_1, b_1, x_2, y_2, b_2, x_3, y_3, b_3, rgb=None, label_1='cluster 1', label_2='cluster 2', label_3='cluster 3', color_1='black', color_2='orange', color_3='blue', returns=True, n_clusters=3):
    
    """
    This function clusters pixels based on brightness using k-means and plots the result.

    It takes the x, y coordinates and the brightness of a pixel as the inputs. As well as some technical inputs
    that determine wether it should return the variables it used in the scatter plot it makes or not. And
    wether the scatter plot function should get the rgb array to plot it as the original picture.

    The function works by taking three groups of encoded galaxy pixels (gray, yellow, blue), computing their
    brightness values, clustering them using K-Means, and then visualizing the resulting clusters in a cropped
    scatter plot, optionally with the RGB galaxy image shown underneath.
    """

    xs = np.concatenate([x_1, x_2, x_3])
    ys = np.concatenate([y_1, y_2, y_3])

    brightness_all = np.concatenate([b_1, b_2, b_3])

    color_codes = np.concatenate([
        np.zeros_like(x_1,    dtype=float),
        np.ones_like(x_2,   dtype=float),
        2 * np.ones_like(x_3, dtype=float),])
    
    X = brightness_all.reshape(-1, 1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(X)

    mask0 = labels == 0
    mask1 = labels == 1
    mask2 = labels == 2

    x_c0, y_c0 = xs[mask0], ys[mask0]
    x_c1, y_c1 = xs[mask1], ys[mask1]
    x_c2, y_c2 = xs[mask2], ys[mask2]

    fig, ax = scatter_plot(x_c0, y_c0, x_c1, y_c1, x_c2, y_c2, rgb, label_1, label_2, label_3, color_1, color_2, color_3)

    if returns:
        return x_c0, y_c0, x_c1, y_c1, x_c2, y_c2
    

def kmeans_cluster_yellowness_and_plot(x_1, y_1, yell_1, x_2, y_2, yell_2, x_3, y_3, yell_3, rgb=None, label_1='cluster 1', label_2='cluster 2', label_3='cluster 3', color_1='black', color_2='orange', color_3='blue', returns=True, n_clusters=3):
    
    """
    This function clusters pixels based on 'yellowness' using k-means and plots the result.

    It takes the x, y coordinates and a yellowness value for each pixel as inputs. It also allows specifying
    custom labels and colors for the three clusters, and can optionally draw the RGB image as a background.

    The function works by taking three groups of encoded galaxy pixels, combining their yellowness values into a
    single array, clustering them using K-Means, and then visualizing the resulting clusters in a cropped scatter
    plot using the 'scatter_plot' function, optionally with the RGB galaxy image shown underneath.
    """

    xs = np.concatenate([x_1, x_2, x_3])
    ys = np.concatenate([y_1, y_2, y_3])

    yellowness_all = np.concatenate([yell_1, yell_2, yell_3])

    color_codes = np.concatenate([np.zeros_like(x_1, dtype=float), np.ones_like(x_2, dtype=float), 2 * np.ones_like(x_3, dtype=float)])

    X = yellowness_all.reshape(-1, 1)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(X)

    mask0 = labels == 0
    mask1 = labels == 1
    mask2 = labels == 2

    x_c0, y_c0 = xs[mask0], ys[mask0]
    x_c1, y_c1 = xs[mask1], ys[mask1]
    x_c2, y_c2 = xs[mask2], ys[mask2]

    fig, ax = scatter_plot(x_c0, y_c0, x_c1, y_c1, x_c2, y_c2, rgb, label_1, label_2, label_3, color_1, color_2, color_3)

    if returns:
        return x_c0, y_c0, x_c1, y_c1, x_c2, y_c2
    


def linear_regression(t, O, country):

    #Calculate cumulative outbreaks and reshape time values for sklearn
    y = np.cumsum(O)
    x = t.reshape(-1, 1)
    
    #Fit line to the data through linear regression
    model= LinearRegression()
    model.fit(x, y)
    
    #Predict the cumulative cases over the gime
    y_pred = model.predict(x)
    
    plt.scatter(x, y, label="Cumulative data")
    plt.plot(x, y_pred, color="red", label="Fitted line")
    plt.title(f"Linear regression of cumulative Ebola cases in {country}")
    plt.xlabel("Days since first outbreak")
    plt.ylabel("Cumulative number of cases")
    plt.legend()
    plt.show()



def polynomial_regression(t, O, country):
    
    #Calculate cumulative outbreaks and reshape time values for sklearn
    y = np.cumsum(O)
    x = np.array(t).reshape(-1, 1)
    
    
    #Creating 3rd degree polynomial features for the curvilinear regression
    poly = PolynomialFeatures(degree=3, include_bias=False)
    x_poly = poly.fit_transform(x)
    
    
    # Fit regression model with polynomial features
    model = LinearRegression()
    model.fit(x_poly, y)
    
    #Predict the cumulative cases over the gime
    y_pred = model.predict(x_poly)

    plt.scatter(t, y, label="Cumulative data")
    plt.plot(t, y_pred, color="red", label=f"Polynomial")
    plt.title(f"Polynomial regression of Ebola cases in {country}")
    plt.xlabel("Days since first outbreak")
    plt.ylabel("Cumulative number of cases")
    plt.legend()
    plt.show()

def nn_pred(t, O, country):
    
    #Calculate cumulative outbreaks and reshape time values for sklearn
    y = np.cumsum(O)
    x = t.reshape(-1, 1)
    
    #Split set for 80% training data and 20% testing data retaining chronological order
    split_index = int(0.8* len(x))
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    
    # Define small neural network with L2 regularization to reduce overfittin on the small dataset
    model = Sequential([
        Dense(64, activation="relu", input_dim=1, kernel_regularizer=l2(0.01)),
        Dense(64, activation="relu", kernel_regularizer=l2(0.01)),
        Dense(1)])
    
    #Compiling the model using mean squared error and Adam optimizer
    model.compile(optimizer="adam", loss="mean_squared_error")
    
    #Training network on the training data
    model.fit(x_train, y_train, epochs=200, verbose=0)
    
    #Prediction of cumulative cases on the test portion of the data
    y_pred = model.predict(x_test)
    
    plt.scatter(t, y, label='Cumulative data')
    plt.plot(t[split_index:], y_pred, color='red', label='NN prediction')
    plt.title(f"Neural network regression of Ebola cases in {country}")
    plt.xlabel("Days since first outbreak")
    plt.ylabel("Cumulative number of cases")
    plt.legend()
    plt.show()
    
    