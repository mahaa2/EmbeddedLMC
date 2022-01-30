println("Running LMC-Monge for the funnel model with D = 2")

# Set the model - Choose
M = funnel(2, 2.0, 15.0);
# number of samples
N = 10000

# call the LMCe function
X = LMCea(M, 2 * randn(M.D), N, 0.13, 15, "0", 1.0);

# Plot the chains together with the countour plots
p1 = plot( X[1, :], alpha = 0.9, label = "x");
     plot!(X[end, :], alpha = 0.9, label = "a");

xl = collect(LinRange(- 1.8 + M.mmean, 1.8 + M.mmean, 100));
yl = collect(LinRange(- 0.6 * M.sig2v - M.mmean, 0.5 * M.sig2v + M.mmean, 100));

p2 = plot(xl, yl, (x, y) -> exp(M.lp(M, [x, y])), xlabel = "x", ylabel = "a",
    st = [:contour], fill = false, legend = :none, left_margin = 2mm, title = "LMC-Monge");
scatter!(X[1, :], X[2, :], xlim = (xl[1], xl[end]),
    ylim = (yl[1], yl[end]), alpha = 0.2);

h1 = histogram(X[1, :], nbins = 40, alpha = 0.9, normalize = true, label = "x");
h2 = histogram(X[2, :], nbins = 40, alpha = 0.9, normalize = true, label = "a");

plot(p2, h1, h2, p1, size = [1000, 700], layout = @layout [grid(1, 3); c{0.5h}])

# # compare with the state-of-the-art method and implementation of nuts and adaptation
# ℓπ(θ) = M.lp(M, θ)
# g(θ) = (ℓπ(θ), M.g(M, θ))
# xini = 2 * randn(M.D)
# hamil = Hamiltonian(DiagEuclideanMetric(M.D), ℓπ, g)
# integrator = Leapfrog(find_good_stepsize(hamil, xini))
# prop = NUTS{MultinomialTS, GeneralisedNoUTurn}(integrator)
# adaptor = StanHMCAdaptor(MassMatrixAdaptor(DiagEuclideanMetric(D)),
#  StepSizeAdaptor(1.0, integrator))
# samp, = sample(hamil, prop, xini, N, adaptor, 5; progress = true);
# Y = hcat(samp...)

# p1 = plot( Y[1, :], alpha = 0.9, label = "x");
#      plot!(Y[end, :], alpha = 0.9, label = "a");

# xl = collect(LinRange(- 1.8 + M.mmean, 1.8 + M.mmean, 100));
# yl = collect(LinRange(- 0.6 * M.sig2v - M.mmean, 0.5 * M.sig2v + M.mmean, 100));

# p2 = plot(xl, yl, (x, y) -> exp(M.lp(M, [x, y])), xlabel = "x", ylabel = "a",
#     st = [:contour], fill = false, legend = :none, left_margin = 2mm, title = "HMC-Nuts");
# scatter!(Y[1, :], Y[2, :], xlim = (xl[1], xl[end]),
#     ylim = (yl[1], yl[end]), alpha = 0.2);

# h1 = histogram(Y[1, :], nbins = 40, alpha = 0.9, normalize = true, label = "x");
# h2 = histogram(Y[2, :], nbins = 40, alpha = 0.9, normalize = true, label = "a");

# plot(p2, h1, h2, p1, size = [1000, 700], layout = @layout [grid(1, 3); c{0.5h}])