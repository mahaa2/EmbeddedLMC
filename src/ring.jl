Base.@kwdef mutable struct ring <: targdist
    D::Union{Int64, Any} = 2
    mean::Float64 = 0.1
    sig2::Float64 = 1.0
    lp::Union{Function, Any} = []
    g::Union{Function, Any} = []
    H::Union{Function, Any} = []
    G::Union{Function, Any} =[]
    iG::Union{Function, Any} = []
    ind::Vector{Int64} = collect(1:3:4)
    cachm::cache
end

function ring(mean::Float64 = 5.0, sig2::Float64 = 1.0, D::Int64 = 2)

    function logp(mod::ring, x::Vector{T}) where T <: Number
        # log-likelihood

        r = sqrt(x[1]^2.0 + x[2]^2.0)
        lp = -0.5 * (log(2.0 * pi) + log(mod.sig2) + (r - mod.mean)^2.0 / mod.sig2)

        return lp - log(r)
    end

    function dlogp(mod::ring, x::Vector{T}) where T <: Number

        mod.cachm.u .= 0.0

        r = sqrt(x[1]^2.0 + x[2]^2.0)
        dlogr = - (r - mod.mean) / mod.sig2

        mod.cachm.u[1] += dlogr * (1.0 / r) * x[1] - x[1] / (r^2.0)
        mod.cachm.u[2] += dlogr * (1.0 / r) * x[2] - x[2] / (r^2.0)

        ind0 = abs.(mod.cachm.u) .< 1e-15
        any(ind0) ? mod.cachm.u[ind0] .= (sign.(mod.cachm.u[ind0]) .* 0.0) : nothing

        mg = maximum(abs.(mod.cachm.u))
        mg > maxintfloat() && isfinite(mg) ? mod.cachm.u .*= 0.3 : mod.cachm.u[isnan.(mod.cachm.u)] .= 0.0

        return mod.cachm.u
    end

    function d2logp(mod::ring, x::Vector{T}) where T <: Number

        mod.cachm.U .= 0.0

        r = sqrt(x[1]^2.0 + x[2]^2.0)

        dlogr = - (r - mod.mean) / mod.sig2 - 1.0 / r
        d2logr = - 1 / mod.sig2 + 1.0 / (r^2.0)

        drdx1 =  x[1] / r
        drdx2 =  x[2] / r

        d2rdx1 = 1.0 / r - x[1]^2.0 / (r^(3.0))
        d2rdx2 = 1.0 / r - x[2]^2.0 / (r^(3.0))

        mod.cachm.U[1, 1] += d2logr * drdx1^2.0 + dlogr * d2rdx1
        mod.cachm.U[1, 2] = mod.cachm.U[2, 1] += d2logr * drdx1 * drdx2 - dlogr * x[1] * x[2]/(r^3.0)
        mod.cachm.U[2, 2] += d2logr * drdx2^2.0 + dlogr * d2rdx2

        ind0 = abs.(mod.cachm.U) .< 1e-15 
        any(ind0) ? mod.cachm.U[ind0] .= (sign.(mod.cachm.U[ind0]) .* 0.0) : nothing

        mH = maximum(abs.(mod.cachm.U))
        mH > maxintfloat() && isfinite(mH) ? mod.cachm.U .*= 0.3 : mod.cachm.U[isnan.(mod.cachm.U)] .= 0.0

        return mod.cachm.U
    end

    function G(mod::ring, x::Vector{T}, α::T = 1.0) where T <: Number
		# metric-tensor embedding

		g = mod.g(mod, x)
		broadcast!(*, mod.cachm.U, g, g');
		mod.cachm.U .*= α^2.0
		mod.cachm.U[mod.ind] .+= 1.0

		return mod.cachm.U
    end

    function iG(mod::ring, x::Vector{T}, α::T = 1.0) where T <: Number
        # inverse metric tensor

		g = mod.g(mod, x)
		L = 1.0 + α^2.0 * norm(g)^2.0
		broadcast!(*, mod.cachm.U, g, g');
		mod.cachm.U .*= - α^2.0 / L
		mod.cachm.U[mod.ind] .+= 1.0

		return mod.cachm.U
    end

    return ring(D, mean, sig2, logp, dlogp, d2logp, G ,iG, collect(1:3:4),
    cache(zeros(D), zeros(D, D), zeros(D), zeros(D), zeros(D)))
end