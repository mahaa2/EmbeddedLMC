# read the data and prepare it
println("Running LMC-Monge for logistic regression with the australian dataset")

data = readdlm("../data/australian.dat");

ind = Colon();
A = data[ind, 1:(end-1)];
A = broadcast(/, broadcast(-, A, mean(A, dims = 1)), std(A, dims = 1));
A = hcat(ones(size(A, 1)), A);
y = data[ind, end];
(N, D) = size(A);

# set the logistic regression model and the data
M = logreg(D, prSparse(D));
M.y = copy(y);
M.A = copy(A);

# run lmce
@time X = LMCea(M, 1e-3 * ones(M.D), 20000, 0.077, 6, "0", 0.01);

# calculate efficient sample
ess = [effective_sample_size(X[i, 5001:end]) for i in 1:M.D]
summarystats(ess)

