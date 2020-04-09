#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This is an implementation of the FH dW model with pymc3. It uses the
same data set for co.nz as https://blog.nzrs.net.nz/domain-retention-prediction/ 

It outputs a plot of simulated retention rates using the the HPD3%, HPD97% 
and MAP estimators for the dW theta and c parameters, together with the
plot line for the underlying data-set.

The sBG model implemented in the NZ blogpost underestimates customer 
retention rates over longer periods of time. Th discrete-Weibull
model released by Peter Fader and Bruce Hardie in 2018 is meant to
compensate for this long-term underestimation problem by introducing
time-variance on the churn probability (i.e. the longer a customer has
stayed, the longer it is likely that they will stay).

The dW model follows the one outlined in  
“How to Project Customer Retention” Revisited: The Role of Duration Dependence
Peter S. Fader, Bruce G.S. Hardie, Yuzhou Liu, Joseph Davin, Thomas Steenburgh
in Journal of Interactive Marketing 43 (2018) 1 – 16


In the below pymc3 mode, the mapValues dictionary of MAP
estimators shows theta and c to be estimated to the same values  
as those obtained by Fader et al in Table 4.

For q = 5 (simulating parameters based on five time-stamps) the 
underestimation problem persists in a fairly severe way for the NZ data.
For q = 10 the underestimation problem also persists but in a less
severe way. 

It remains to be seen whether longer time-series can give better long-term
estimations. The discrete-Weibull estimation seems, in either case,
to give a better fit for NZ data than the Beta-dW estimation does.

"""

import numpy as np
from matplotlib import pyplot as plt
import pymc3 as pm
import theano.tensor as tt


# Example data from NZ
exampleData = np.array([3712, 3031, 2318, 1889, 1667, 1498, 1357, 1246, 1163,
                         1107, 1055, 996, 935])

# Example data from Fader and Hardie-paper (High end customers)
#exampleData = np.array([1000, 869, 743, 653, 593, 551, 517, 
#                        491, 468, 445, 427, 409, 394])

def nLost(data):
    lost = [0]
    for i in range(1, len(data)):
        lost.append(data[i - 1] - data[i])
    return lost

def nRetained(data):
    retained = [1] * len(data)
    for i in range(1, len(data)):
        retained[i] = data[i]/data[i-1]
    return retained


q = 5

exampleDataNLost = nLost(exampleData)

n = len(exampleData[0:q])
dataFive = np.asarray((exampleData[0:q], exampleDataNLost[0:q]))

with pm.Model() as BdWwithcfromUnif:
    theta = pm.Uniform('theta',0.0001, 1.0, testval=0.5)
    c = pm.Uniform('c', 0.0001, 1000.0, testval=1.0)

    p = [0.]*n
    s = [1.]*n
    for t in range(1, n):
        s[t] = (1-theta)**(t**c)
        p[t] = s[t-1]-s[t]
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

    retention = pm.DensityDist('retention', logp, observed=dataFive)
    step = pm.DEMetropolis()
    trace = pm.sample(10000, step=step, tune=1000)

# Maximum a posteriori estimators for the model
mapValues = pm.find_MAP(model=BdWwithcfromUnif)

theta = mapValues.get('theta').item()
cHat = mapValues.get('c').item()
#rvar = stats.beta.var(betaParams[0], betaParams[1])

thetaUp = pm.summary(trace).get('hpd_97%').get('theta').item()
cHatUp = pm.summary(trace).get('hpd_97%').get('c').item()

thetaLow = pm.summary(trace).get('hpd_3%').get('theta').item()
cHatLow = pm.summary(trace).get('hpd_3%').get('c').item()

exampleDataRetention = nRetained(exampleData)

# Define a Discrete Weibull distribution
def DiscreteWeibull(q, b, x):
    return (1-q)**(x**b) - (1-q)**((x+1)**b)

# Plot stuff
x = np.linspace(0,len(exampleData)-1,len(exampleData))
# Distribution also shifted one step to the right.
weibDist = DiscreteWeibull(theta, cHat, x+1)

weibDistSorted = np.sort(weibDist)[::-1]
weibDistRetained = nRetained(weibDistSorted/weibDistSorted[0])

weibDistUp = DiscreteWeibull(thetaUp, cHatUp, x+1)

weibDistSortedUp = np.sort(weibDistUp)[::-1]
weibDistRetainedUp = nRetained(weibDistSortedUp/weibDistSortedUp[0])

weibDistLow = DiscreteWeibull(thetaLow, cHatLow, x+1)
weibDistSortedLow = np.sort(weibDistLow)[::-1]
weibDistRetainedLow = nRetained(weibDistSortedLow/weibDistSortedLow[0])
# Plot
plt.plot(x, np.array(exampleDataRetention))
plt.plot(x, weibDistRetained)
plt.plot(x, weibDistRetainedUp)
plt.plot(x, weibDistRetainedLow)
plt.show()