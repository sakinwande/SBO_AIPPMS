mutable struct GaussianProcess
    m # mean
    mXq # mean function at query points
    k # covariance function
    X # design points
    X_query # query points (assuming these always stay the same)
    y # objective values
    ν # noise variance
    KXX # K(X,X) the points we have measured
    KXqX # K(Xq,X) the points we are querying and we have measured
    KXqXq
end

μ(X, m) = [m(x) for x in X]

Σ(X, k) = kernelmatrix(k, X, X)
K(X, X′, k) = kernelmatrix(k, X, X′)

function mvnrand(rng, μ, Σ, inflation=1e-6)
    N = MvNormal(μ, Σ + inflation*I)
    return rand(rng, N)
end
Base.rand(rng, GP, X) = mvnrand(rng, μ(X, GP.m), Σ(X, GP.k))
Base.rand(rng, GP, μ_calc, Σ_calc) = mvnrand(rng, μ_calc, Σ_calc)

function query_no_data(GP::GaussianProcess)
    μₚ = GP.mXq
    S = GP.KXqXq
    νₚ = diag(S) .+ eps() # eps prevents numerical issues
    return (μₚ, νₚ, S)
end

function query(GP::GaussianProcess)
    # tmp = GP.KXqX / (GP.KXX + diagm(GP.ν))
    tmp = GP.KXqX / (GP.KXX + Diagonal(GP.ν))
    μₚ = GP.mXq + tmp*(GP.y - μ(GP.X, GP.m))
    S = GP.KXqXq - tmp*GP.KXqX'
    νₚ = diag(S) .+ eps() # eps prevents numerical issues
    return (μₚ, νₚ, S)
end

function posterior(GP::GaussianProcess, X_samp, y_samp, ν_samp)
    if GP.X == []
        KXX = kernelmatrix(GP.k, X_samp, X_samp)
        KXqX = kernelmatrix(GP.k, GP.X_query, X_samp)

        return GaussianProcess(GP.m, GP.mXq, GP.k, X_samp, GP.X_query, y_samp, ν_samp, KXX, KXqX, GP.KXqXq)
    else
        a = kernelmatrix(GP.k, GP.X, X_samp)
        KXX = [GP.KXX a; a' 1.0] #KXX = [GP.KXX a; a' I]
        KXqX = [GP.KXqX kernelmatrix(GP.k, GP.X_query, X_samp)]#hcat(GP.KXqX, kernelmatrix(k, GP.X_query, X_samp)

        return GaussianProcess(GP.m, GP.mXq, GP.k, [GP.X; X_samp], GP.X_query, [GP.y; y_samp], [GP.ν; ν_samp], KXX, KXqX, GP.KXqXq)

    end
end
