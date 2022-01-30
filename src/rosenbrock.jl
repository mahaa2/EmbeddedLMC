Base.@kwdef mutable struct rosenbr <: targdist
    D::Union{Int64, Any} = 2
    a::Float64 = 0.1
    b::Float64 = 40.0
    lp::Union{Function, Any} = []
    g::Union{Function, Any} = []
    H::Union{Function, Any} = []
    G::Union{Function, Any} = []
    iG::Union{Function, Any} = []
    ind::Vector{Int64} = collect(1:3:4)
    cachm::cache
end

function rosenbr(a::Float64 = 0.1, b::Float64 = 40.0, D::Int64 = 2)

    function logp(mod::rosenbr, x::Vector{T}) where T <: Number
        return -(mod.a - x[1])^2.0 - mod.b * (x[2] - x[1]^2.0)^2.0
    end

    function dlogp(mod::rosenbr, x::Vector{T}) where T <: Number
        
        mod.cachm.u .= 0.0

        mod.cachm.u[1] +=  2.0 * (mod.a - x[1]) + 4.0 * mod.b * (x[2] - x[1]^2.0) * x[1]
        mod.cachm.u[2] += -2.0 * mod.b * (x[2] - x[1]^2.0)

        ind0 = abs.(mod.cachm.u) .< 1e-15
        any(ind0) ? mod.cachm.u[ind0] .= (sign.(mod.cachm.u[ind0]) .* 0.0) : nothing

        mg = maximum(abs.(mod.cachm.u))
        mg > maxintfloat() && isfinite(mg) ? mod.cachm.u .*= 0.3 : mod.cachm.u[isnan.(mod.cachm.u)] .= 1.0

        return mod.cachm.u
    end

    function d2logp(mod::rosenbr, x::Vector{T}) where T <: Number
        
        mod.cachm.U .= 0.0

        mod.cachm.U[1, 1] += -2.0 + 2.0 * mod.b * (-6.0 * x[1]^2.0 + 2.0*x[2])
        mod.cachm.U[2, 1] = mod.cachm.U[1, 2] += 4.0 * mod.b * x[1]
        mod.cachm.U[2, 2] += -2.0 * mod.b

        ind0 = abs.(mod.cachm.U) .< 1e-15 
        any(ind0) ? mod.cachm.U[ind0] .= (sign.(mod.cachm.U[ind0]) .* 0.0) : nothing

        mH = maximum(abs.(mod.cachm.U))
        mH > maxintfloat() && isfinite(mH) ? mod.cachm.U .*= 0.3 : mod.cachm.U[isnan.(mod.cachm.U)] .= 1.0

        return mod.cachm.U
    end

    function G(mod::rosenbr, x::Vector{T}) where T <: Number
		# metric-tensor embedding

		g = mod.g(mod, x)
		broadcast!(*, mod.cachm.U, g, g');
		mod.cachm.U[mod.ind] .+= 1.0

		return mod.cachm.U
    end

    function iG(mod::rosenbr, x::Vector{T}) where T <: Number
        # inverse metric tensor

		g = mod.g(mod, x)
		L = 1.0 + norm(g)^2.0
		broadcast!(*, mod.cachm.U, g, g');
		mod.cachm.U ./= - L;
		mod.cachm.U[mod.ind] .+= 1.0

		return mod.cachm.U
    end

    return rosenbr(D, a, b, logp, dlogp, d2logp, G, iG, collect(1:(D+1):D^2), 
    cache(zeros(D), zeros(D, D), zeros(D), zeros(D), zeros(D)))
end
