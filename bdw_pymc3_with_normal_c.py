#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This is an implementation of the FH BdW model with pymc3. It uses the
same data set for co.nz as https://blog.nzrs.net.nz/domain-retention-prediction/ 

The sBG model implemented in the NZ blogpost underestimates customer 
retention rates over longer periods of time. The Beta-discrete-Weibull
model released by Peter Fader and Bruce Hardie in 2018 is meant to
compensate for this long-term underestimation problem by introducing
time-variance on the churn probability (i.e. the longer a customer has
stayed, the longer it is likely that they will stay).

The BdW model follows the one outlines in Annex B of 
“How to Project Customer Retention” Revisited: The Role of Duration Dependence
Peter S. Fader, Bruce G.S. Hardie, Yuzhou Liu, Joseph Davin, Thomas Steenburgh
in Journal of Interactive Marketing 43 (2018) 1 – 16

In the below pymc3 model with q=5, the mapValues dictionary of MAP
estimators shows that alpha, beta and c are estimated to exactly those 
values obtained by Fader et al in Table 4. Yet generating a dW variable
from these parameters does not yield as good of a fit with the actual
underlying data as the one  obtained by Fader et al *unless* the parameter 
c is divided by 2.

In the below model, c is assumed to be normally distributed. Default mu=0.5
since we assume the default behaviour of a consumer is to be more likely
to not cancel a contract the longer they have already stuck around.

Larger values for q do not mitigate under-estimation in the long-term. 

TODO: implement without hyper-parametrised theta.

"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats
import pymc3 as pm
import theano.tensor as tt

# Example data from NZ
#exampleData = np.array([3712, 3031, 2318, 1889, 1667, 1498, 1357, 1246, 1163,
#                         1107, 1055, 996, 935])

# Example data from Fader and Hardie-paper (High end customers)
exampleData = np.array([1000, 869, 743, 653, 593, 551, 517, 
                        491, 468, 445, 427, 409, 394])
def nLost(data):
    lost = [0]
    for i in range(1, len(data)):
        lost.append(data[i - 1] - data[i])
    return lost

q = 10

exampleDataNLost = nLost(exampleData)

n = len(exampleData[0:q])
data = np.asarray((exampleData[0:q], exampleDataNLost[0:q]))

with pm.Model() as BdWwithcfromNorm:
    alpha = pm.Uniform('alpha',0.0001, 1000.0, testval=1.0)
    beta = pm.Uniform('beta', 0.0001, 1000.0, testval=1.0)
    # If c=1, the dW collapses to a geometric distribution. We assume
    # that in a usual case, customer survival probability normally
    # stay the same over time.
    # If mu = 1.5 we get exactly the same sampling results as if
    # we had used a uniform prior. The result is slightly better than
    # defining a N(1,None) variable.
    c = pm.Normal('c', mu=0.5, testval=1.0)

    p = [0.]*n
    s = [1.]*n
    logB = tt.gamma(alpha+beta)/tt.gamma(beta)
    for t in range(1, n):
        pt1 = tt.gamma(beta+(t-1)**c)/tt.gamma(beta+alpha+(t-1)**c)
        pt2 = tt.gamma(beta+t**c)/tt.gamma(beta+alpha+t**c)
        s[t] = pt2*logB
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

    retention = pm.DensityDist('retention', logp, observed=data)
    step = pm.DEMetropolis()
    trace = pm.sample(10000, step=step, tune=2000)

# Maximum a posteriori estimators for the model
mapValues = pm.find_MAP(model=BdWwithcfromNorm)

# Extract alpha and beta MAP-estimators
betaParams = mapValues.get('alpha').item(), mapValues.get('beta').item()

theta = stats.beta.mean(betaParams[0], betaParams[1])
cHat = mapValues.get('c').item()
rvar = stats.beta.var(betaParams[0], betaParams[1])

# Define a Discrete Weibull distribution
def DiscreteWeibull(q, b, x):
    return (1-q)**(x**b) - (1-q)**((x+1)**b)

# Plot stuff
x = np.linspace(0,len(exampleData)-1,len(exampleData))
# For some reason this is only a good fit when cHat is divided by 2.
# Distribution also shifted one step to the right.
weibDistrR = DiscreteWeibull(theta, cHat/2, x+1)
# Upper bound and normalized
weibDistrRup = (weibDistrR + rvar)*(exampleData[0]/(weibDistrR[0]+rvar))
# Lower bound and normalized. <0 values removed.
weibDistrRdown = (weibDistrR - rvar)*(exampleData[0]/(weibDistrR[0]-rvar))
weibDistrRdown = [0 if weibDistrRdown_ < 0 
                  else weibDistrRdown_ for weibDistrRdown_ in weibDistrRdown]
weibDistrRmid = weibDistrR*(exampleData[0]/weibDistrR[0])
# Plot
plt.plot(x, exampleData)
plt.plot(x, weibDistrRmid, 'b-')
plt.plot(x, weibDistrRup, 'b--')
plt.plot(x, weibDistrRdown, 'b--')
plt.show()
