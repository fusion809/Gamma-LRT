using StatsFuns, DecFP, CSV, DataFrames, Statistics, SpecialFunctions
using LinearAlgebra

function getVars(group, y)
    m = maximum(group)
    nvec = zeros((m, 1))
    for i=1:m
        nvec[i, 1] = length(y[isequal.(group, i)])
    end
    nvec = Int.(nvec)
    ni = maximum(nvec)
    yarr = BigFloat.(zeros((m, ni)))
    
    for i=1:m
        for j=1:nvec[i, 1]
            yarr[i,j] = y[isequal.(group, i)][j]
        end
    end

    ybarvec = mean(yarr, dims=2)
    ybarvec = reshape(ybarvec, m, 1)
    alphavec = 10 * ones((m, 1))

    return m, ni, alphavec, nvec, yarr, ybarvec
end

function funjacUnr(alphavec, nvec, yarr, ybarvec)
    Jinv = Diagonal(vec((nvec .* (alphavec.^(-1)-polygamma.(1, Float64.(alphavec)))).^(-1)))
    logsum = reshape(sum(log.(yarr), dims=2), (m, 1))
    F = - nvec .* (polygamma.(0, Float64.(alphavec)) + log.(ybarvec .* alphavec.^(-1))) + logsum
    return Jinv, F
end

function funjacNull(alpha, n, yarr, ybar)
    J = n * (1/alpha - polygamma(1, Float64(alpha)))
    F = -n * (polygamma(0, Float64(alpha)) + log(ybar/alpha)) + sum(log.(yarr))
    return J, F
end

function newtonsUnr(m, alphavec, nvec, yarr, ybarvec)
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

function newtonsNull(alpha, n, yarr, ybar)
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

csv_reader = CSV.File("ProjectData.csv")
dataF = DataFrame(csv_reader)
y = BigFloat.(dataF[:, 6])
group = dataF[:, 1]
m, ni, alphavec, nvec, yarr, ybarvec = getVars(group, y)
n = length(y)
ybar = mean(y)
alphavec = newtonsUnr(m, alphavec, nvec, yarr, ybarvec)
betavec = ybarvec/alphavec
alpha = 1
alpha = newtonsNull(alpha, n, yarr, ybar)
beta = ybar/alpha

lam = (gamma(alpha) * (ybar/alpha)^(alpha))^(-n)
lam *= prod(prod(yarr.^(alpha*ones(length(alphavec), 1)-alphavec), dims=2))
lam *= prod((gamma.(alphavec) .* (ybarvec.*alphavec.^(-1)).^(alphavec)).^(nvec))
stat = -2*log(lam)
pval = 1-chisqcdf(2*m-2, Float64(stat))