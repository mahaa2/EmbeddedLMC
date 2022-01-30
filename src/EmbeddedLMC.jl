
module EmbeddedLMC

 using SpecialFunctions,
       Distributions,
       LinearAlgebra,
       QuadGK

 abstract type LMCE end
 abstract type targdist <: LMCE end
 abstract type prior <: LMCE end
 
 export LMCE,
        targdist,
        prior,
        LMCea,
        cache,
        logreg2,
        rosenbrock,
        bansh,
        squiggle,
        funnel,
        priorSparse,
        ring

 # ─ include LMC algorithm and models
 include("LMCea.jl")
 include("cache.jl")
 include("logreg2.jl")
 include("rosenbrock.jl")
 include("bansh.jl")
 include("squiggle.jl")
 include("funnel.jl")
 include("priorSparse.jl")
 include("ring.jl")
 
 println("LMC algorithm with monge-metric and functionalitites loaded")
end

# ─ extra packages
using AdvancedHMC,
      Plots,
      StatsPlots,
      MCMCDiagnostics,
      Random,
      StatsBase,
      DelimitedFiles