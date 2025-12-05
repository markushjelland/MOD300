import matplotlib.pyplot as plt
import numpy as np

def ODE_Solver(t0, t_final, y0, dt, f, beta_function, b0, l): # Implementing the ODE solver:

    N = int(t_final/dt) # Defining the amount of generations.
    
    t = [t0] # Adding the initial conditions as first elements in lists.
    y = [y0]

    for i in range(N):
        t_old = t[-1] # Defining old and new time values.
        t_new = t_old + dt
        
        beta = beta_function(b0, t_new, l) # Defining beta based on current time.

        """
        The use of the beta function is due to excercise 3, task 2.
        When beta is a constant, we really don't need it, but it is usefull later.
        """

        y_old = y[-1] # Defining old and new y values.
        y_new = y_old + RK4(y_old, t_old, dt, f, beta) # y_new is found using the Runge-Kutta 4. order method ('RK4' function).

        t.append(t_new) # Saving the new state of the system in my lists.
        y.append(y_new)

    return np.array(t), np.array(y) # returning arrays containing every state of the sytem in "t" and "y" dimensions respectively.


def RK4(y, t, dt, f, beta): # Defining the RK4 algorithm:
    k1 = np.array(f(y, t, beta))
    k2 = np.array(f(y + dt*0.5*k1, t + 0.5*dt, beta))
    k3 = np.array(f(y + dt*0.5*k2, t + 0.5*dt, beta))
    k4 = np.array(f(y + dt*k3, t + dt, beta))

    return (dt/6)*(k1 + 2*k2 + 2*k3 + k4) # returning the step y(dt), not y(t0) + y(dt).

#Skipping the header and dates to simply data extraction
guinea  = np.loadtxt("ebola_cases_guinea.dat", skiprows=1, usecols=(1, 2))
liberia  = np.loadtxt("ebola_cases_liberia.dat", skiprows=1, usecols=(1, 2))
sierra  = np.loadtxt("ebola_cases_sierra_leone.dat", skiprows=1, usecols=(1, 2))

def ebola_plot(t, O, country):
    """
    Plots reported Ebola data for the given country.
    
    Input
    -----
    t: array
        Number of days since first outbreak
    O: array
        Reported daily outbreaks
    country: str
        Name of country to be plotted
    
    Output:
    -----
    figure: plot
        Daily and cumulative outbreaks over days since first outbreak
    """
    
    
    cumulative = np.cumsum(O)
    
    #Plotting daily daily outbreaks on left y-axis and cumulative outbreaks on right y-axis
    figure, axis1 = plt.subplots()
    axis1.plot(t, O, "o", markeredgecolor="r", markerfacecolor="none")
    axis1.set_xlabel("Days since first outbreak")
    axis1.set_ylabel("Number of outbreaks")
    axis1.grid(True)
    
    axis2 = axis1.twinx()
    axis2.plot(t, cumulative, "s", markeredgecolor="k", markerfacecolor="none")
    axis2.set_ylabel("Cumulative number of outbreaks", color="k")

    
    plt.title(f"Ebola outbreaks in {country}")
    plt.show()





#Split into days elapsed and cumulative outbreaks
t_g, O_g = guinea[:, 0], guinea[:, 1]
t_l, O_l = liberia[:, 0], liberia[:, 1]
t_s, O_s = sierra[:, 0], sierra[:, 1]


#ebola_plot(t_g, O_g, "Guinea")
#ebola_plot(t_l, O_l, "Liberia")
#ebola_plot(t_s, O_s, "Sierra Leone")


def beta_exp(beta0, t, lam):
    """
    transmission rate for Ebola: beta(t) = beta0 * exp(-lam * t)
    
    Input
    -----
    beta0: float
        Initial contact rate at t = 0
    t: float
        Days since first reported outbreak
    lam: float
        Rate of reduction in beta
    """
    return beta0 * np.exp(-lam * t)

def ebola_RHS(y, t, beta, sigma=1/9.7, gamma=1/7):
    """
    SEZR model for Ebola outbreak
    
    Input
    -----
    y: array
        Current state [S, E, Z, R]
    t: float
        Time in days
    beta: float
        Transmission rate at time t
    sigma: float
        Rate of transition from exposed to infected
    gamma: float
        rate of removal/recovery
        
    Output
    -----
    dydt: array
        [dS, dE, dZ, dR]
    """
    
    S, E, Z, R = y
    dS = -beta * S * Z
    dE =  beta * S * Z - sigma * E
    dZ =  sigma * E - gamma * Z
    dR =  gamma * Z
    
    return np.array([dS, dE, dZ, dR])

def ebola_model_plot():
    
    #Initial conditions
    g_N = 1e7
    g_E0 = 0.0
    g_Z0 = O_g[0]/g_N
    g_R0 = 0.0
    g_S0 = 1.0 - g_Z0
    g_y0 = np.array([g_S0, g_E0, g_Z0, g_R0])


    #Adjustable model parameters
    beta0 = 0.3445
    lam   = 0.0032

    #Simulation time range and step size
    t0, T, dt = 0.0, float(t_g[-1]), 7.0
    t_model, y_model = ODE_Solver(t0, T, g_y0, dt, ebola_RHS, beta_exp, beta0, lam)


    #Extracting compartments
    S_mod = y_model[:, 0]
    E_mod = y_model[:, 1]

    #Calculating daily and cumulative cases from model output
    daily_model = g_N * (1/9.7 * E_mod)     
    cumulative_model = g_N * (1.0 - S_mod) 

    figure, axis1 = plt.subplots()
    axis1.plot(t_model, daily_model, "o", markeredgecolor="r", markerfacecolor="none")
    axis1.set_xlabel("Days since first outbreak")
    axis1.set_ylabel("Number of outbreaks")
    axis1.grid(True)

    axis2 = axis1.twinx()
    axis2.plot(t_model, cumulative_model, "s", markeredgecolor="k", markerfacecolor="none")
    axis2.set_ylabel("Cumulative number of outbreaks", color="k")

    plt.title("Model of Ebola outbreaks in Guinea")
    plt.show()
