# Lagrangian Manifold Monte Carlo on Monge Patches 

This computer program provides a reference implementation of Lagrangian Monte Carlo in metric induced by the Monge patch.
The code was prepared to the final version of the accepted manuscript in AISTATS and is provided as-is. 

Requirements :
 1. The code has been tested on Julia version 1.6.3, but is likely to work on all recent versions.
 2. The code relies on few packages that can be installed in Julia REPL using	

    'add SpecialFunctions, Distributions, LinearAlgebra, Plots, StatsPlots, AdvancedHMC, MCMCDiagnostics, Random, StatsBase, DelimitedFiles, QuadGK

Using the code :
 1. Type 'include("EmbeddedLMC.jl")' to install the module EmbeddedLMC and include all required functionalitites
 2. After that type 'using .EmbeddedLMC'
 3. The main algorithm is provided in LMCea.jl as the function LMCea() that takes 8 input arguments:
	- 1st argument is the target distribution
	- 2nd argument is the initial value of the mcmc chain
	- 3rd argument is the sample-size of the chain
	- 4th argument is the step-size of the numerical integrator
	- 5th argument is the number of leapfrog steps
	- 6th argument should be '0' (experimental functionality for step-size adaptation)
	- 7th argument is the value of \alpha 
	- 8th argument is a given initial velocity vector (for examples); if it is not given then the velocity vector will be sampled from a multivariate Gaussian 

 4. There are 7 probabilistic models which can be used. They are 
	- "bansh.jl" The banana-shaped probability distribution from Lan et. al. 2015 (Markov Chain Monte Carlo from Lagrangian Dynamics) 
	- "rosenbrock.jl" Another banana-shaped distribution obtained from the rosenbrock function
	- "squiggle.jl" the same probabilistic model from https://chi-feng.github.io/mcmc-demo/
	- "funnel.jl" The classic funnel distribution from Radford Neal
	- "priorSparse.jl" Generalized Gaussian distribution 
	- "logreg2.jl" binary regression with the logistic link function 
	- "ring.jl" A probabilistic distribution where the typical set has a form of a ring on R^2
 
 5. Files example-funnel.jl, example-logreg.jl, example-squiggle provide examples on how to use the code

