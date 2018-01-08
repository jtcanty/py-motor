
# A library for simulating time-series data with jump points #

import numpy as np
import pandas as pd
import random
import math
import statsmodels.api as sm
from scipy.optimize import curve_fit

def _logL(trace, curr_steps, n_pts, n_steps):
    ''' Calculate the minimum SIC statistic for all models with n steps. 
    
    Parameters
    ----------
    
    Returns
    -------
    sic_min : float
        The lowest SIC score
        
    step_idx: int
        Index of step that yielded the lowest SIC score 
        
    '''
    sic_fit = np.zeros(n_pts)
    
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
        sic_fit[step] = sic
        
    # Compute minimum SIC score and step position
    mask = (sic_fit != 0)
    trunc_step_idx = np.argmin(sic_fit[mask])
    step_idx = np.arange(sic_fit.shape[0])[mask][trunc_step_idx]
    sic_min = sic_fit[step_idx]

    return (sic_min, step_idx)
   

def _merge_steps(trace_noisy, curr_steps):
    '''Construct the optimal SIC fit and plot it. Output the fit coordinates'''
    fit_split = np.split(trace_noisy, np.sort(curr_steps))
    dwell_means = np.array([np.mean(fit_split[i]) for i in range(0, len(fit_split))])
    dwell_fit = [dwell_means[j] * np.ones(len(fit_split[j])) for j in range(0, len(fit_split))]
    fit_func = np.concatenate(dwell_fit)

    return fit_func   


class Simulation(object):
    ''' Generates time-series data with discrete jump-points.

    Attributes
    ----------
    pos_mu : float
        The mean of the positive jump-point distance.
        Units should be 'nm'.

    pos_sigma : float
        The standard deviation of the mean positive jump-point 
        distance. Units should be 'nm'.

    neg_mu : float
        The mean of the negative jump-point distance.
        Units should be 'nm'.

    neg_sigma : float
        The standard deviation of the mean negative jump-point 
        distance. Units should be 'nm'.

    pos_freq : float
        The expected proportion of positive jump points.

    neg_freq : float
        The expected proportion of negative jump points.

    k : float
        The expected jump-point frequency. Units should be 's^-1'.
    
    rate: float
        The time resolution of the time series signal.
        Units should be 's'

    GN_sigma : float
        The expected gaussian white noise (GN) of the time-series signal.
        Units should be 'nm'.

    n_steps : int
        Number of expected steps in the time-series.

    num_trace : int
        Number of expected traces to be simulated.

    '''
    def __init__(self, 
                 pos_mu=12, 
                 pos_sigma=3,
                 neg_mu=12, 
                 neg_sigma=3,
                 fwd_freq=.8, 
                 bkwd_freq=.2,
                 k=1, 
                 rate=.03,
                 GN_sigma=4, 
                 n_steps=30,
                 num_trace=1):
       
        # Initialize step-size parameters
        self.pos_mu = pos_mu
        self.pos_sigma = pos_sigma
        self.neg_mu = neg_mu
        self.neg_sigma = neg_sigma

        # Initialize step frequency parameters         
        self.fwd_freq = fwd_freq
        self.bkwd_freq = bkwd_freq
        
        # Initialize remaining parameters         
        self.k = k
        self.rate = rate
        self.sigma = GN_sigma
        self.n_steps = n_steps  
    
    
    def build(self):
        '''The main loop for building a single time-series trace

        Parameters
        ----------
        self : object
            The time-series simulation object.

        Returns
        -------
        trace_noisy : array, shape = [n_points]
            The time vs. displacement coordinates of the simulated 
            time series including the expected white noise.

        trace_ideal : array, shape = [n_points]
            The time vs. displacement coordinates of the simulated
            time series without added white-noise.
                
        '''
        trace_noisy = np.array([0])
        trace_ideal = np.array([0])

        for step in range(0, self.n_steps):

            # Construct dwell from exponential distribution
            dwell_len = random.expovariate(1 / self.k)
            dwell_pts = int(dwell_len // self.rate)

            GN = np.random.normal(0, self.GN_sigma, dwell_pts)
            dwell_noise = np.full(dwell_pts, trace_noisy[-1]) + GN
            trace_noisy = np.append(trace_noisy, dwell_noise)

            dwell_ideal = np.full(dwell_pts, trace_ideal[-1])
            trace_ideal = np.append(trace_ideal, dwell_ideal)
            
            # Randomly decide step direction. Sample step-size from normal distribution.
            pr = np.random.random_sample()

            if pr > self.bkwd_freq:
                step_dist = random.gauss(self.pos_mu, self.pos_sigma)
            
            elif pr <= self.bkwd_freq:
                step_dist = -random.gauss(self.neg_mu, self.neg_sigma)

            trace_ideal[-1] += step_dist
            trace_noisy[-1] += step_dist     
            step += 1

        # Add final dwell
        dwell_len = random.expovariate(1 / self.k)
        dwell_pts = int(dwell_len // self.rate)

        GN = np.random.normal(0, self.GN_sigma, dwell_pts)
        dwell = np.full(dwell_pts, trace_noisy[-1]) + GN
        trace_noisy = np.append(trace_noisy, dwell)

        dwell_ideal = np.full(dwell_pts, trace_ideal[-1])
        trace_ideal = np.append(trace_ideal, dwell_ideal)

        return (trace_ideal, trace_noisy)
    
    
    def fit(self, trace_noisy):
        '''Fit the simulated traces using the SIC step-fitting algorithm

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
        fit: array
            A numpy array of coordinates corresponding to the optimal SIC fit 

        '''
        curr_steps = []
        min_threshold = 1

        n_steps = 0
        n_pts = len(trace_noisy)

        # Calculate initial SIC value
        SIC = (n_steps + 2) * np.log(n_pts) + n_pts * np.log(np.var(trace_noisy)) + n_pts

        while True:
            # Compute SIC and add new step
            (SIC_next, step_new_idx) = _logL(trace_noisy, curr_steps, 
                                                    n_pts, n_steps)
            if SIC >= SIC_next:
                curr_steps.append(step_new_idx)
                SIC = SIC_next
                continue
                
            elif SIC < SIC_next:
                break
        
        # Assemble steps into a trace fit
        fit = _merge_steps(trace_noisy, curr_steps)

        return (fit, curr_steps)

    
    
    def assemble_steps(ntraces, fits):
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

    
    def assemble_dwells(ntraces, step_location, rate):
        '''Assemble dwells of all step locations

        Parameters
        ----------
        ntraces: int
            Number of traces
        step_location: pandas DataFrame
            DataFrame of binary arrays where 1s correspond to step locations
        rate: int
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
        all_dwells_converted = [i*rate for i in all_dwells]
        return all_dwells_converted


    def stats(all_steps, all_dwells):
        '''Compute important statistics of the time-series.

        Parameters
        ----------
        all_steps :  list 
            A list of jump-point sizes. Units are 'nm'.
            
        all_dwells :  list
            A List of dwell times betwen jump-points.
            Units are 's'.

        Returns
        -------
        stats: list, shape = [8,]
            A list of the parameters listed below.
            
        fwd_mu: float
            Mean of forward step-size

        bkwd_mu: float
            Mean of backwards step-size

        fwd_std: float
            Standard deviation of forwards step-size

        bkwd_std: float
            Standard deviation of backwards step-size
            
        fwd_freq: float
            Frequency of forward steps
            
        bkwd_freq: float
            Frequency of backward steps
            
        dwell_k: float
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
