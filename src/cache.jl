Base.@kwdef mutable struct cache <: LMCE
	u::Union{Float64, Vector{Float64}}    = zeros(2)
	U::Union{Float64, Matrix{Float64}}    = zeros(2, 2)
    ∇lTH::Union{Float64, Vector{Float64}} = zeros(2)
    vTH::Union{Float64, Vector{Float64}}  = zeros(2)
    Hv::Union{Float64, Vector{Float64}}   = zeros(2)
end

