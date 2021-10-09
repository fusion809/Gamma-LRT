using StatsFuns, DecFP, CSV, DataFrames, Statistics, SpecialFunctions
using LinearAlgebra

"""
    getVars(group::Vector, y::Vector)

Computes various required variables from the grouping (group) and dependent 
variables (y). Includes m, number of groups; n, total number of observations; 
ni, maximum sample size; alphavec, m x 1 matrix of our initial guess for 
``\\alpha_i``; nvec, m x 1 matrix of sample sizes for each group; yarr, 
m x ni matrix of observations; ybar, the overall mean for all observations;
and ybarvec, m x 1 matrix of group means.
"""
function getVars(group, y)
    m = maximum(group)
    nvec = zeros((m, 1))
    for i=1:m
        nvec[i, 1] = length(y[isequal.(group, i)])
    end
    nvec = Int.(nvec)
    n = length(y)
    ni = maximum(nvec)
    yarr = BigFloat.(zeros((m, ni)))
    
    for i=1:m
        for j=1:nvec[i, 1]
            yarr[i,j] = y[isequal.(group, i)][j]
        end
    end

    ybar = mean(y)
    ybarvec = mean(yarr, dims=2)
    ybarvec = reshape(ybarvec, m, 1)
    alphavec = 10 * ones((m, 1))

    return m, n, ni, alphavec, nvec, yarr, ybar, ybarvec
end

"""
    funjacUnr(alphavec::Matrix, nvec::Matrix, yarr::Matrix, ybarvec::Matrix)

Computes and returns the inverse of the Jacobian and the matrix of function 
values for computing the unrestricted MLE of alpha_i.
"""
function funjacUnr(alphavec, nvec, yarr, ybarvec)
    Jinv = Diagonal(vec((nvec .* (alphavec.^(-1)-polygamma.(1, Float64.(alphavec)))).^(-1)))
    logsum = reshape(sum(log.(yarr), dims=2), (m, 1))
    F = - nvec .* (polygamma.(0, Float64.(alphavec)) + log.(ybarvec .* alphavec.^(-1))) + logsum
    return Jinv, F
end

"""
    funjacNull(alpha::Float64, n::Int64, yarr::Matrix, ybar::Matrix)

Computes and returns the derivative and function value required to compute
the MLE of alpha under the null hypothesis.
"""
function funjacNull(alpha, n, yarr, ybar)
    J = n * (1/alpha - polygamma(1, Float64(alpha)))
    F = -n * (polygamma(0, Float64(alpha)) + log(ybar/alpha)) + sum(log.(yarr))
    return J, F
end

"""
    newtonsUnr(m::Int64, alphavec::Matrix, nvec::Matrix, yarr::Matrix, 
    ybarvec::Matrix, itMax::Int64, tol::Float64)

Estimates the unrestricted MLE of alpha_i using Newton's method.
"""
function newtonsUnr(m, alphavec, nvec, yarr, ybarvec, itMax, tol)
    Jinv, F = funjacUnr(alphavec, nvec, yarr, ybarvec)
    eps = -Jinv * F
    diff = sqrt(sum(eps.^2)/m)
    iteration = 0
    itMax = 1e3
    tol = 1e-13

    while ((tol < diff) && (iteration < itMax))
        alphavec += eps
        Jinv, F = funjacUnr(alphavec, nvec, yarr, ybarvec)
        eps = -Jinv * F
        epsRel = eps / alphavec
        diff = sqrt(sum((epsRel^2)/m))
        iteration += 1
    end

    return alphavec
end

"""
    newtonsNull(alpha::Float64, n::Int64, yarr::Matrix, ybar::Float64, 
    itMax::Int64, tol::Float64)

Estimates the MLE of alpha under the null using Newton's method.
"""
function newtonsNull(alpha, n, yarr, ybar, itMax, tol)
    J, F = funjacNull(alpha, n, yarr, ybar)
    eps = -F/J
    
    iteration = 0
    itMax = 1e3
    tol = 1e-13

    while ((tol < eps/alpha) && (iteration < itMax))
        alpha += eps
        J, F = funjacNull(alpha, n, yarr, ybar)
        eps = -F/J
        iteration +=1
    end
    
    return alpha
end

# Get problem data and parameters
csv_reader = CSV.File("ProjectData.csv")
dataF = DataFrame(csv_reader)
y = BigFloat.(dataF[:, 6])
group = dataF[:, 1]
m, n, ni, alphavec, nvec, yarr, ybar, ybarvec = getVars(group, y)

# Estimate unrestricted MLEs
alphavec = newtonsUnr(m, alphavec, nvec, yarr, ybarvec)
betavec = alphavec.^(-1) .* ybarvec

# Estimate MLEs under null
alpha = 1
alpha = newtonsNull(alpha, n, yarr, ybar)
beta = ybar/alpha

# Likelihood ratio
lam = (gamma(alpha) * (ybar/alpha)^(alpha))^(-n)
lam *= prod(prod(yarr.^(alpha*ones(length(alphavec), 1)-alphavec), dims=2))
lam *= prod((gamma.(alphavec) .* (ybarvec.*alphavec.^(-1)).^(alphavec)).^(nvec))

# Test statistic, -2 ln(lambda)
stat = -2*log(lam)

# Obtain p-value keeping in mind that under the null our test statistic
# should asymptotically follow a chi-squared distribution with 2m-2 df
pval = 1-chisqcdf(2*m-2, Float64(stat))

# Printing important data
println("alpha (null)   = ", Float64(alpha))
println("beta (null)    = ", Float64(beta))
println("alpha_i        = ", Float64.(alphavec))
println("beta_i         = ", Float64.(betavec))
println("Lambda         = ", lam)
println("Test statistic = ", Float64(stat))
println("P-value        = ", pval)