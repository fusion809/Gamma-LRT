#!/usr/bin/env python
import numpy as np
import csv
from scipy.stats import chi2
from scipy.special import polygamma, gamma

# Read the most important pieces of data from the CSV
def readData(fileName, groupNo, depVarNo):
    """
    Returns group variable and dependent variable in fileName that are in the
    columns specified by groupNo and depVarNo. 

    Parameters
    ----------
    fileName : string.
               The CSV file we're reading data from.
    groupNo  : int.
               An integer indicating the column in fileName in which our 
               grouping variable is.
    depVarNo : int.
               An integer indicating the column in fileName in which our
               dependent variable is.

    Returns
    -------
    group    : NumPy array of integers.
               Contains the grouping variable for each observation.
    y        : NumPy array of floats.
               Contains the dependent variable value for each observation.
    """
    # Initialize variables for reading from file
    ifile = open(fileName)
    reader = csv.reader(ifile)

    # Initialize arrays to store variable data
    y = np.array([])
    group = np.array([])

    # Initialize count for loop below
    count = 0

    # Loop through the rows in input file
    for row in reader:
        if ((depVarNo >= np.size(row)) or (groupNo >= np.size(row))):
            print("At least one of the specified column numbers depVarNo or")
            print("groupNo are greater than or equal to the number of columns")
            print("in specified file.")
            exit()
        # Do not include headers
        if (count != 0):
            group = np.append(group, int(row[groupNo]))
            y = np.append(y, float(row[depVarNo]))
        count += 1

    ifile.close()

    return group, y

def getVars(group, y):
    """
    Calculate various variables we need from group and y.

    Parameters
    ----------
    group   : NumPy array of ints.
              Group variable corresponding to each observation.
    y       : NumPy array of floats.
              Dependent variable value corresponding to each observation.

    Returns
    -------
    m       : int.
              Number of groups.
    muNull  : NumPy array of floats.
              Initial estimate of our MLE for the mean under the null.
    varNull : NumPy array of floats.
              Initial estimate of our MLE for the variance under the null.
    nvec    : NumPy array of ints.
              Vector of sample sizes for each value of the grouping variable.
    yarr    : NumPy array of floats.
              Array of values of the dependent variable for each observation 
              with each row corresponding to a different value of the grouping
              variable.
    ybarvec : NumPy array of floats.
              Means of the dependent variable for each value of the grouping 
              variable.
    """
    # Number of groups
    m = int(np.max(group))

    # Vector of sample sizes
    nvec = np.tile(0, m)
    for i in range(0, m):
        nvec[i] = int(np.size(y[group == i+1]))

    nvec = np.reshape(nvec, (m, 1))

    # Maximum sample size
    ni = int(np.max(nvec))

    # Initialize 2D array for storing y values categorized by treatment group
    yarr = np.tile(0.5, (m, ni))

    # yarr rows correspond to different groups
    # columns different observations
    for i in range(0, m):
        for j in range (0, nvec[i,0]):
            yarr[i, j] = y[group == i + 1][j]

    # Ybar_i
    ybarvec = np.mean(yarr, axis=1)
    ybarvec = np.reshape(ybarvec, (m, 1))

    # Initial guess of mu under the null
    alphavec = 10*np.ones((m, 1))

    return m, ni, alphavec, nvec, yarr, ybarvec

def funjacUnr(alphavec, nvec, yarr, ybarvec):
    """
    Return inverse of Jacobian and function vector for unrestricted MLE 
    problem.

    Parameters
    ----------
    alphavec : NumPy array of floats.
               Our current estimate of the MLE of alpha_i.
    nvec     : NumPy array of ints.
               Contains sample sizes for each group.
    yarr     : NumPy array of floats.
               Array of observations, each row corresponds to different groups.
    ybarvec  : NumPy array of floats.
               Array of means for each treatment group.
    
    Returns
    -------
    Jinv     : NumPy array of floats.
               Inverse of Jacobian.
    F        : NumPy array of floats.
               Function values array.
    """
    Jvec = nvec * (1/alphavec - polygamma(1, alphavec))
    Jinv = np.diagflat(1/Jvec)
    #Jinv = np.diag(Jinv)
    logsum = np.reshape(np.sum(np.log(yarr), axis=1), (m, 1))
    F = -nvec * ( polygamma(0, alphavec) + np.log(ybarvec/alphavec)) + logsum

    return Jinv, F

def funjacNull(alpha, n, yarr, ybar):
    """
    Return Jacobian and function vector for null MLE problem.

    Parameters
    ----------
    alpha : float.
            Initial guess of the MLE of alpha under the null.
    n     : int.
            Total number of observations.
    yarr  : NumPy array of floats.
            Array of observations with each row corresponding to different 
            groups.
    ybar  : float.
            Mean of y. 

    Returns
    -------
    J     : float.
            Function derivative.
    F     : float.
            Function value. 
    """
    J = n * (1/alpha - polygamma(1, alpha))
    F = -n*(polygamma(0, alpha) + np.log(ybar/alpha)) + np.sum(np.log(yarr))
    return J, F

def newtonsUnr(m, alphavec, nvec, yarr, ybarvec, itMax, tol):
    """
    Approximate unrestricted MLE of alpha_i using Newton's method.

    Parameters
    ----------
    m        : int.
               Number of groups. 
    alphavec : NumPy array of floats.
               Initial guess of the MLE of alpha_i.
    nvec     : NumPy array of ints.
               Array of sizes for each sample (group).
    yarr     : NumPy array of floats.
               Array of observations with different rows corresponding to
               different groups.
    ybarvec  : NumPy array of floats.
               Array of means for each group.
    itMax    : int or float.
               Maximum number of iterations of Newton's method allowed when 
               estimating our MLE.
    tol      : float.
               Relative error tolerance.
    """
    Jinv, F = funjacUnr(alphavec, nvec, yarr, ybarvec)
    eps = -np.matmul(Jinv, F)
    epsRel = eps / alphavec
    diff = np.sqrt(np.sum(epsRel**2)/m)

    iteration = 0
    
    while (tol < diff and iteration < itMax):
        alphavec += eps
        Jinv, F = funjacUnr(alphavec, nvec, yarr, ybarvec)
        eps = -np.matmul(Jinv, F)
        epsRel = eps / alphavec
        diff = np.sqrt(np.sum(epsRel**2)/m)
        iteration += 1

    print("Number of iterations used to approximate alphavec = {}".format(iteration))
    return alphavec

def newtonsNull(alpha, n, yarr, ybar, itMax, tol):
    """
    Approximate MLE of alpha under the null hypothesis using Newton's method.

    Parameters
    ----------
    alpha : float.
            Our initial estimate of alpha.
    n     : int.
            Total number of observations.
    yarr  : NumPy array of floats.
            All observations arranged in a m x max(nvec) array.
    ybar  : float.
            Mean of all observations.
    itMax : int or float.
            Maximum number of iterations can be used to estimate alpha.
    tol   : float.
            Relative error tolerance.
    
    Returns
    -------
    alpha : float.
            Refined estimate of alpha using Newton's method.
    """
    J, F = funjacNull(alpha, n, yarr, ybar)
    eps = -F/J
    
    iteration = 0

    while (tol < np.abs(eps)/alpha and iteration < itMax):
        alpha += eps
        J, F = funjacNull(alpha, n, yarr, ybar)
        eps = -F/J
        iteration += 1

    print("Number of iterations used to approximate alpha = {}".format(iteration))
    return alpha

# alphavec's 2nd element is -inf when using OutlierRm
#group, y = readData("ProjectDataOutlierRm.csv", 0, 4)
# alphavec is fine when using original data set
# But lam is nan due to gamma(alphavec) failing due to floating
# point arithmetic limitations
group, y = readData("ProjectData.csv", 0, 5)
m, ni, alphavec, nvec, yarr, ybarvec = getVars(group, y)
n = np.size(y)
ybar = np.mean(yarr)
itMax = 1e3
tol = 1e-13
alphavec = newtonsUnr(m, alphavec, nvec, yarr, ybarvec, itMax, tol)
betavec = ybarvec/alphavec
alpha = 1
alpha = newtonsNull(alpha, n, yarr, ybar, itMax, tol)
beta = ybar / alpha

# For my dataset there is an alphavec entry = 260.2153579
# Gamma(260.2153579) > 1e508, too big for ufuncs like scipy.special.gamma
# to handle
# I get the error:
# TypeError: ufunc 'gamma' not supported for the input types, and the inputs 
# could not be safely coerced to any supported types according to the casting 
# rule ''safe''
# When I convert alphavec to float128 type before running gamma on it
lam = np.power(1/(gamma(alpha)*(ybar/alpha)**(alpha)), n)
lam = lam * np.prod(np.power((gamma(alphavec) * np.power(ybarvec/alphavec, 
alphavec)), nvec))
lam *= np.prod(np.prod(np.power(yarr, alpha-alphavec), axis=1))

# Test statistic
stat = -2*np.log(lam)

# P-value
pval = 1 - chi2.cdf(stat, 2*m-2)

# Equivalent test for exponential distribution
# Under null, all groups share the same exponential distribution parameter.
# Under alternative hypothesis, at least two groups have different exponential
# distribution parameters.
statExp = 2*n*np.log(ybar) - 2*np.sum(nvec * np.log(ybarvec))
pvalExp = 1 - chi2.cdf(statExp, m-1)