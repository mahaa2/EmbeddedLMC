println("Running LMC-Monge and NUTS for the squiggle probabilistic model")

# Random.seed!(2)
npoints = 300
xran = LinRange(-11, 11, npoints)
yran = LinRange(-1.5, 1.5, npoints)

aD = [0.5, 1.0, 2.0]
modD = [squiggle(a, [6.0 0.01; 0.01 0.001]) for a in aD]
N = 60000
X = Array{Array{Float64, 2}, 1}(undef, length(aD))
Y = Array{Array{Float64, 2}, 1}(undef, length(aD))
xini = [1.0, -1.0]

for i in 1:length(aD)
    M = modD[i]

    # Lagrangian MC embedded graph metrics
    if i == 1 || i == 2
        @time X[i] = LMCea(M, xini, N, 0.07, 13, "0", 1.0);
    else
        @time X[i] = LMCea(M, xini, N, 0.025, 36, "0", 1.0);
    end

    # HMC-NUTS
    ℓπ(θ) = M.lp(M, θ)
    g(θ) = (ℓπ(θ), M.g(M, θ))
    hamil = Hamiltonian(DiagEuclideanMetric(M.D), ℓπ, g)
    integrator = Leapfrog(find_good_stepsize(hamil, xini))
    prop = NUTS{SliceTS, ClassicNoUTurn}(integrator)
    samp = sample(hamil, prop, xini, N; progress = true)
    Y[i] = hcat(samp[1]...)
end

# supplementary figures
pd = Array{Plots.Plot, 1}(undef, length(aD));
pl = Array{Plots.Plot, 1}(undef, length(aD));

for i2 = 1:length(aD)
    # density plots
    pd[i2] = plot(xran, yran, (x, y) -> exp(modD[i2].lp(modD[i2], [x, y])), 
    st = [:contour], levels = 100, fill = false, 
    legend = :none, title = "a = $(aD[i2])" )

    # marginal distribution comparison 
    fX = x -> modD[i2].marg(modD[i2], x)[1]

    # plot
    pl[i2] = plot(xlims = (-11, 11), fX, label = "", ticks = nothing, w = 2.0)
    histogram!(X[i2][1, 5000:end], nbins = 60, alpha = 0.4, normalize = true, 
    label = "LMC-Monge, a = $(aD[i2])")
    histogram!(Y[i2][1, 5000:end], nbins = 60, alpha = 0.4, normalize = true,
    label = "HMC-Nuts, a = $(aD[i2])")
    
    plot!(X[i2][1, 5000:end], alpha = 0.4, label = "LMC-M",
    inset = (1, bbox(0.01, 0.15, 0.35, 0.35, :top, :left)),
    subplot = 2, ticks = nothing, w = 1.1)
    plot!(Y[i2][1, 5000:end], alpha = 0.4, label = "HMC-Nuts", 
    ticks = nothing, w = 1.1, subplot = 2)
end

p5 = plot(pd[1], pd[2], pd[3], pl[1], pl[2], pl[3], size = [1700, 900], 
thickness_scaling = 1.2, dpi = 220, layout = (2, 3))
