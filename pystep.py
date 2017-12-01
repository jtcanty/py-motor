
# A python implementation of simulated molecular motor traces #

import numpy as np
import pandas as pd
import random
import math
import statsmodels.api as sm
from scipy.optimize import curve_fit



def _log_likelihood(trace, curr_steps, n_pts, n_steps):
    ''' Calculate the SIC statistic for all models with i steps. 
    
    Parameters
    ----------
    
    Returns
    -------
    sic_min : float
        The lowest SIC score
        
    step_idx: int
        Index of step that yielded the lowest SIC score 
        
    '''
    sic_array = np.zeros(n_pts)
    
    for step in range(0,n_pts):
        if step in curr_steps:
            continue
        
        if step == 0:
            continue
            
        # Set new step at current point and calculate global variance of all dwells.
        temp = curr_steps + [step]
        trace_split = np.split(trace,np.sort(temp))
        dwell_variance = [np.var(trace_split[j]) * len(trace_split[j]) for j in range(0,len(trace_split))]
        variance_est = (1/n_pts) * sum(dwell_variance)

        # Compute SIC score
        n_steps = len(trace_split) - 1
        sic = (n_steps + 2) * np.log(n_pts) + n_pts * np.log(variance_est)
        sic_array[step] = SIC
        
    # Compute minimum SIC score and step position
    mask = (sic_array != 0)
    trunc_step_idx = np.argmin(sic_array[mask])
    step_idx = np.arange(sic_array.shape[0])[mask][trunc_step_idx]
    sic_min = sic_array[step_idx]

    return (sic_min, step_idx)
   


class Simulation(object):
    '''This class defines a simulation object
    
    Parameters
    ----------
    
    Attributes
    ----------
    
    ''' 
    def __init__(self, 
                 fwd_step_mu=12, 
                 fwd_step_sigma=3,
                 bkwd_step_mu=12, 
                 bkwd_step_sigma=3,
                 fwd_freq=.8, 
                 bkwd_freq=.2,
                 step_rate=.1, 
                 framerate=.002,
                 sigma_noise=4, 
                 num_steps=30,
                 num_trace=1):
        '''

        Attributes
        ----------
        fwd_step_mu : int
            Mean of the forward step-size (nm)
            
        fwd_step_sigma : int
            Standard deviation of the forward step-size (nm)
            
        bkwd_step_mu : int
            Mean of the backward step-size (nm)
            
        bkwd_step_sigma : int
            Standard deviation of the backward step-size (nm)
            
        fwd_freq : float
            Frequency of forward steps
            
        bkwd_freq : float
            Frequency of backward steps 
            
        step_rate : float
            Stepping rate (sec^-1)
            
        framerate : float
            Time resolution of the experiment (sec)
            
        sigma_noise : int
            White noise of the time-series (nm)
            
        num_steps : int
            Number of expected steps in the time-series
            
        num_trace : int
            Number of expected traces to be simulated
            
        Returns
        -------
        self : class instance
            Instance of the Simulation class
            
        '''
        # Initialize step-size parameters
        self.fwd_step_mu = fwd_step_mu
        self.fwd_step_sigma = fwd_step_sigma
        self.bkwd_step_mu = bkwd_step_mu
        self.bkwd_step_sigma = bkwd_step_sigma

        # Initialize step frequency parameters         
        self.fwd_freq = fwd_freq
        self.bkwd_freq = bkwd_freq
        
        # Initialize remaining parameters         
        self.step_rate = step_rate
        self.framerate = framerate
        self.sigma_noise = sigma_noise
        self.num_steps = num_steps  
    
    
    def build(self):
        '''The main loop for building a single-molecule trace

            Parameters
            ----------
            

            Returns
            -------
            trace_noisy: np.array
                Coordinates of a simulated trace
                
            trace_ideal: np.array
                Coordinates of simulated trace without noise
                
        '''
        trace_noisy = np.array([0])
        trace_ideal = np.array([0])

        for step in range(0, self.num_steps):

            # Create dwell
            dwell_length = random.expovariate(1 / self.step_rate)
            dwell_pts = int(dwell_length // self.framerate)

            noise = np.random.normal(0, self.sigma_noise, dwell_pts)
            dwell_noise = np.full(dwell_pts, trace_noisy[-1]) + noise
            trace_noisy = np.append(trace_noisy, dwell_noise)

            dwell_ideal = np.full(dwell_pts, trace_ideal[-1])
            trace_ideal = np.append(trace_ideal, dwell_ideal)
            
            # Add next step. Roll dice to decide step direction and then sample gaussian pdf for step-size
            pb = np.random.random_sample()

            if pb > self.bkwd_freq:
                step_dist = random.gauss(self.fwd_step_mu, self.fwd_step_sigma)
            elif pb <= self.bkwd_freq:
                step_dist = -random.gauss(self.bkwd_step_mu, self.bkwd_step_sigma)

            trace_ideal[-1] += step_dist
            trace_noisy[-1] += step_dist     
            step += 1

        # Add final dwell
        dwell_length = random.expovariate(1 / self.step_rate)
        dwell_pts = int(dwell_length // self.framerate)

        noise = np.random.normal(0, self.sigma_noise, dwell_pts)
        dwell = np.full(dwell_pts, trace_noisy[-1]) + noise
        trace_noisy = np.append(trace_noisy, dwell)

        dwell_ideal = np.full(dwell_pts, trace_ideal[-1])
        trace_ideal = np.append(trace_ideal, dwell_ideal)

        return (trace_ideal, trace_noisy)
    
    
    def fit(self, trace_noisy):
        '''Main function to fit the simulated traces using the SIC step-fitting algorithm

        Parameters
        ----------
        trace_noisy: np.array
            A numpy array of coordinates corresponding to simulated trace
        curr_steps: list
            Array of indices of the recorded steps
        min_threshold: int
            Minimum number of points to be considered a dwell
        SIC_curr: float
            Current minimized SIC value of fit

        Returns 
        ------- 
        fit: np.array
            A numpy array of coordinates corresponding to the optimal SIC fit 

        '''
        curr_steps = []
        min_threshold = 1

        n_steps = 0
        n_pts = len(trace_noisy)

        # Calculate initial SIC value
        sic_curr = (n_steps + 2) * np.log(n_pts) + n_pts * np.log(np.var(trace_noisy)) + n_pts

        while True:
            # Compute SIC and add new step
            (sic_new, step_new_idx) = logLikelihood(trace_noisy, curr_steps, n_pts, n_steps)
            if sic_curr >= sic_new:
                curr_steps.append(step_new_idx)
                sic_curr = sic_new
                continue
            elif sic_curr < sic_new:
                break

        fit = plot(trace_noisy, curr_steps)

        return (fit,curr_steps)

    
    def plot(trace_noisy, curr_steps):
        '''Construct the optimal SIC fit and plot it. Output the fit coordinates
        
        Parameters
        ----------
        
        Returns
        -------
        fit_func: np.array
            A numpy array of the fit coordinates 
            
        '''
        fit_split = np.split(trace_noisy,np.sort(curr_steps))
        dwell_means = np.array([np.mean(fit_split[i]) for i in range(0,len(fit_split))])
        dwell_fit = [dwell_means[j] * np.ones(len(fit_split[j])) for j in range(0,len(fit_split))]
        fit_func = np.concatenate(dwell_fit)

        return fit_func
    
    
    def steps(ntraces, fits):
        '''Create array of all step-sizes. Create DataFrame of binary arrays for step locations.

        Parameters
        ----------
        ntraces: int
            Number of traces
        fits: pandas DataFrame
            DataFrame of coordinates of either idealized trace or fit

        Returns
        -------
        all_steps: list
            List of all steps. Each element is a step-size in (nm)
        step_location: pandas DataFrame
            A DataFrame of binary arrays where 1s correpsond to step locations

        '''  
        # Calculate all steps
        all_steps = []
        step_location = pd.DataFrame()

        for i in range(0,ntraces):
            fit = fits.iloc[:,i]

            # Add steps to array
            fit_steps = [fit[j] - fit[j-1] for j in range(1,len(fit)) if fit[j] - fit[j - 1] != 0
                             and math.isnan(fit[j] - fit[j - 1]) != True]
            all_steps += fit_steps

            # Create binary array of step locations
            fit_step_locs = [1 if (fit[j] - fit[j - 1] != 0 and math.isnan(fit[j] - fit[j - 1]) != True) else 0 
                                 for j in range(1,len(fit))]
            step_location['{0}'.format(i)] = fit_step_locs

        return (all_steps,step_location)

    
    def dwells(ntraces, step_location, framerate):
        '''Assemble dwells of all step locations

        Parameters
        ----------
        ntraces: int
            Number of traces
        step_location: pandas DataFrame
            DataFrame of binary arrays where 1s correspond to step locations
        framerate: int
            Time resolution of the simulation (in seconds)

        Returns
        -------
        all_dwells_converted: list
            List of integers corresponding to length of a dwell (in seconds)
            
        '''

        all_dwells = []

        for i in range(0,ntraces):
            fit_binary = step_location.iloc[:,i]

            dwells = np.split(fit_binary,np.where(fit_binary == 1)[0])
            all_dwells += [len(dwells[j]) for j in range(0,len(dwells))]

        # Convert from frames to seconds
        all_dwells_converted = [i*framerate for i in all_dwells]
        return all_dwells_converted


    def calculate_statistics(all_steps, all_dwells):
        '''Calculate stepping statistics

        Parameters
        ----------
        all_steps:  list 
            List of step sizes (in nm)
        all_dwells:  list
            List of dwell times (in seconds)

        Returns
        -------
        stats: list
            List containing following variables
            
        fwd_mu:    float64
            Mean of forward step-size

        bkwd_mu:   float64
            Mean of backwards step-size

        fwd_std:   float64
            Standard deviation of forwards step-size

        bkwd_std:  float64
            Standard deviation of backwards step-size
            
        fwd_freq:  float64
            Frequency of forward steps
            
        bkwd_freq: float64
            Frequency of backward steps
            
        dwell_k: float64
            Dwell-time distribution rate constant 
            
        dwell_params: list

        '''
        # Calculate stepping parameters
        fwd_steps = [i for i in all_steps if i >= 0]
        bkwd_steps = [i for i in all_steps if i < 0]

        fwd_mu = np.mean(fwd_steps)
        bkwd_mu = np.mean(bkwd_steps)

        fwd_std = np.std(fwd_steps)
        bkwd_std = np.std(bkwd_steps)

        fwd_freq = len(fwd_steps) / (len(fwd_steps) + len(bkwd_steps))
        bkwd_freq = 1 - fwd_freq

        # Calculate dwell parameters
        ecdf = sm.distributions.ECDF(all_dwells)
        x_c = ecdf.x[1:]
        y_c = 1 - ecdf.y[1:]

        def func(x, a, b, c):
            return a * np.exp(-b * x) + c

        popt, pcov = curve_fit(func, x_c, y_c, p0=[1,1,0])
        f = popt[0] * np.exp(-popt[1] * x_c) + popt[2]
        dwell_constant = 1/popt[1]

        dwell_params = [x_c,y_c,f]
        stats = [fwd_mu, fwd_std, bkwd_mu, bkwd_std, fwd_freq, bkwd_freq, dwell_constant]
        
        return (stats,dwell_params)
