#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This file implements the FH sBG model with pymc3. It uses the
same data set for co.nz as https://blog.nzrs.net.nz/domain-retention-prediction/ 

And follows

https://discourse.pymc.io/t/shifted-beta-geometric-sbg-distribution-in-pymc3/490

The parameter estimation results are similar as in the NZ model. Below
listed for three values of q with NZ-model estimators in parentheses.

6-year: alpha-hat ~ 1.360 (1.221), beta-hat ~ 5.037 (4.491)
7-year: alpha-hat ~ 1.1849 (1.098), beta-hat ~ 4.326 (3.986)
8-year: alpha-hat ~ 1.087 (1.018), beta-hat ~ 3.927 (3.651)

Beta-Geometric distribution underestimates customer retention over long
time-periods. 
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import pymc3 as pm
import theano.tensor as tt

# Example data from NZ
exampleData = np.array([3712, 3031, 2318, 1889, 1667, 1498, 1357, 1246, 1163,
                         1107, 1055, 996, 935])

# Example data from Fader and Hardie-paper (High end customers)
#exampleData = np.array([1000, 869, 743, 653, 593, 551, 517, 
#                        491, 468, 445, 427, 409, 394])

# Function to find # lost customers per time-period
def nLost(data):
    lost = [0]
    for i in range(1, len(data)):
        lost.append(data[i - 1] - data[i])
    return lost

# Number of years used to estimate parameters for model
q = 7

exampleDataNLost = nLost(exampleData)
n = len(exampleData[0:q])

# Array with observed data
data = np.asarray((exampleData[0:q], exampleDataNLost[0:q]))

# Definition of the model
with pm.Model() as sBG:
    alpha = pm.Uniform('alpha',0.0001, 1000.0, testval=1.0)
    beta = pm.Uniform('beta',0.0001, 1000.0, testval=1.0)

    # Allocate an entire array to P(T=t|alpha, beta)
    p = [0.] * n
    p[1] = alpha / (alpha + beta)
    # Allocate an entire array to S(T=t|alpha, beta)
    s = [1.] * n
    s[1] = 1 - p[1]
    for t in range(2, n):
        p[t] = ((beta + t - 2) / (alpha + beta + t - 1)) * p[t-1]
        s[t] = s[t-1] - p[t]
    p = tt.stack(p, axis=0)
    s = tt.stack(s, axis=0)
    
    # Log-likelihood function
    def logp(data):
        observedRenewed = data[0,:]
        observedReleased = data[1,:]

        # Released entries every year
        released = tt.mul(p[1:].log(), observedReleased[1:])

        # Renewed entries every year
        renewed = s[-1].log() * observedRenewed[-1]
        return  released.sum() + renewed

    retention = pm.DensityDist('retention', logp, observed=data)
    step = pm.Metropolis()
    trace = pm.sample(10000, step=step, tune=1000)

# Maximum a posteriori estimators for the model
mapValues = pm.find_MAP(model=sBG)

# Extract alpha and beta MAP-estimators
betaParams = mapValues.get('alpha').item(), mapValues.get('beta').item()

# Generate values from beta-distribution
r = stats.beta.median(betaParams[0], betaParams[1])
rvar = stats.beta.var(betaParams[0], betaParams[1])

# Plot stuff
x = np.linspace(0,len(exampleData)-1,len(exampleData))
# Geometric distribution is shifted one step to the right
# This is the "shifted" part of shifted Beta-Geometric
geomDistrR = stats.geom.pmf(x+1, r)
# Upper confidence interval
geomDistrRup = (geomDistrR + rvar)*(100.0/(geomDistrR[0]+rvar))
# Lower confidence interval
geomDistrRdown = (geomDistrR - rvar)*(100.0/(geomDistrR[0]-rvar))
# Remove values lower than zero since they don't make sense for a pmf
geomDistrRdown = [0 if geomDistrRdown_ < 0 
                  else geomDistrRdown_ for geomDistrRdown_ in geomDistrRdown]
# Normalize
geomDistrRmid = geomDistrR*(100.0/geomDistrR[0])
# Plot
plt.plot(x, exampleData*(100.0/exampleData[0]))
plt.plot(x, geomDistrRmid, 'g-')
plt.plot(x, geomDistrRup, 'g--')
plt.plot(x, geomDistrRdown, 'g--')
plt.show()