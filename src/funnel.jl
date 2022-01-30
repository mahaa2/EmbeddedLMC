Base.@kwdef mutable struct funnel <: targdist
    D::Union{Int64, Any} = 2
    mmean::Float64 = 0.0
    sig2v::Float64 = 0.3
    lp::Union{Function, Any} = []
    g::Union{Function, Any} = []
    H::Union{Function, Any} = []
    G::Union{Function, Any} = []
    iG::Union{Function, Any} = []
    ind::Vector{Int64} = collect(1:3:4)
    cachm::cache
end

function funnel(D::Int64 = 2, mmean::Float64 = 0.0, sig2v::Float64 = 0.3)

    if D <= 1
        error("dimension must be greater than 1")
    end

    function logp(mod::funnel, x::Vector{T}) where T <: Number
        s2 = mod.sig2v
        muth = mod.mmean

        v = x[end]
        sig2th = log(1.0 + exp(v))
        isinf(sig2th) ? sig2th = v : nothing

        D1 = mod.D - 1

        lp = ( - (D1 / 2.0) * (log(2.0 * pi) + log(sig2th)) - 
            0.5 * norm(x[1:(end-1)] .- muth)^2.0 / sig2th -
            0.5 * (log(2.0 * pi) + log(s2) + v^2.0 / s2) )

        return lp
    end

    function dlogp(mod::funnel, x::Vector{T}) where T <: Number

        mod.cachm.u .= 0.0

        s2 = mod.sig2v
        muth = mod.mmean
        D1 = (mod.D - 1)
        
        v = x[end]
        sig2th = log(1.0 + exp(v))
        isinf(sig2th) ? sig2th = v : nothing
        sig2th < eps() ? sig2th = 1e-13 : nothing

        dsig2th = 1.0 / (1.0 + exp(-v)) # exp(v) / (1.0 + exp(v))

        mod.cachm.u[1 : D1] = - (x[1:(end-1)] .- muth) ./ sig2th
        mod.cachm.u[end] = ( - D1 / 2.0 * dsig2th / sig2th +
                     0.5 * norm(x[1:(end-1)] .- muth)^2.0 / sig2th^2.0 * dsig2th -
                     v / s2 )
        
        ind0 = abs.(mod.cachm.u) .< eps()
        any(ind0) ? mod.cachm.u[ind0] .= (sign.(mod.cachm.u[ind0]) .* 0.0) : nothing
             
        mg = maximum(abs.(mod.cachm.u))
        # mg > 1e6 && isfinite(mg) ? mod.cachm.u ./= (0.8 * mg) : nothing
        mg > maxintfloat() && isfinite(mg) ? mod.cachm.u .*= 0.3 : mod.cachm.u[isnan.(mod.cachm.u)] .= 0.0

        return mod.cachm.u
    end

    function d2logp(mod::funnel, x::Vector{T}) where T <: Number
        
        mod.cachm.U .= 0.0

        s2 = mod.sig2v
        muth = mod.mmean
        D1 = (mod.D - 1)
        ind = 1 : D1

        v = x[end]
        sig2th = log(1.0 + exp(v))
        isinf(sig2th) ? sig2th = v : nothing
        sig2th < eps() ? sig2th = 1e-12 : nothing

        dsig2th = 1.0 / (1.0 + exp(-v)) # exp(v) / (1.0 + exp(v))
        d2sig2th = dsig2th - dsig2th^2.0
        # d2sig2th = dsig2th - exp(2.0 * v) / (1.0 + exp(v))^2.0

        d1 = d2sig2th / sig2th - dsig2th^2.0 / sig2th^2.0
        d2 = d2sig2th / sig2th^2.0 - 2.0 * dsig2th^2.0 / sig2th^3.0

        mod.cachm.U[mod.ind] += ( vcat(-ones(D - 1) ./ sig2th,
                        - D1 / 2.0 * d1 
                        + d2 * 0.5 * norm(x[ind] .- muth)^2.0 - 1.0 / s2) )
                        
        mod.cachm.U[end, ind] = mod.cachm.U[ind, end] = (x[ind] .- muth) * dsig2th / sig2th^2.0

        ind0 = abs.(mod.cachm.U) .< eps()
        any(ind0) ? mod.cachm.U[ind0] .= (sign.(mod.cachm.U[ind0]) .* 0.0) : nothing

        mH = maximum(abs.(mod.cachm.U))
        # mH > 1e15 && isfinite(mH) ? mod.cachm.U ./= (0.8 * mH) : nothing
        mH > maxintfloat() && isfinite(mH) ? mod.cachm.U .*= 0.3 : mod.cachm.U[isnan.(mod.cachm.U)] .= 0.0

        return mod.cachm.U
    end

    function G(mod::funnel, x::Vector{<: Number}, α::Number = 1.0)
		# metric-tensor embedding

		g = mod.g(mod, x)
		broadcast!(*, mod.cachm.U, g, g');
		mod.cachm.U .*= α^2.0
		mod.cachm.U[mod.ind] .+= 1.0

		return mod.cachm.U
    end

    function iG(mod::funnel, x::Vector{<: Number}, α::Number = 1.0)
        # inverse metric tensor

		g = mod.g(mod, x)
		L = 1.0 + α^2.0 * norm(g)^2.0
		broadcast!(*, mod.cachm.U, g, g');
		mod.cachm.U .*= - α^2.0 / L
		mod.cachm.U[mod.ind] .+= 1.0

		return mod.cachm.U

    end

    return funnel(D, mmean, sig2v, logp, dlogp, d2logp,
            G, iG, collect(1:(D+1):D^2),
            cache(zeros(D), zeros(D, D), zeros(D), zeros(D), zeros(D)))
end
