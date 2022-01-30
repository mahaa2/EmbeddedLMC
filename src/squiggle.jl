Base.@kwdef mutable struct squiggle <: targdist
    D::Union{Int64, Any} = 2
	a::Float64 = 5.0;
    lp::Union{Function, Any} = []
    g::Union{Function, Any} = []
    H::Union{Function, Any} = []
	G::Union{Function, Any} = []
	iG::Union{Function, Any} = []
    MM::Union{MvNormal, Any} = []
    marg::Union{Function, Any} = []
    ind::Vector{Int64} = collect(1:3:4)
	cachm::cache
end

function squiggle(a::Float64 = 5.0, Sig::Matrix{Float64} = [2.0 0.25; 0.25 0.05])
    D = 2
    mu = zeros(D)
    dist = MvNormal(mu, Sig)

    function marg(mod::squiggle, x::Number) 		
        return quadgk(y -> exp(logpdf(mod.MM, [x, y + sin(mod.a * x)])), -15, 15) 
    end

    function logp(mod::squiggle, x::Vector{T}) where T <: Number
		mod.cachm.u .= 0.0

        mod.cachm.u[1] += x[1]
        mod.cachm.u[2] += x[2] + sin(mod.a * x[1])
		
        return logpdf(mod.MM, mod.cachm.u)
    end

    function dlogp(mod::squiggle, x::Vector{T}) where T <: Number

		mod.cachm.u .= 0.0

        ~isfinite(x[1]) ? x[1] = sign(x[1]) * maxintfloat() : nothing
        ~isfinite(x[2]) ? x[2] = sign(x[2]) * maxintfloat() : nothing

        y = zeros(D)
        y[1] = x[1]
        y[2] = x[2] + sin(mod.a * x[1]) 

        gy = gradlogpdf(mod.MM, y)

		mod.cachm.u[1] += gy[1] + gy[2] * mod.a * cos(mod.a * x[1]);
		mod.cachm.u[2] += gy[2]

		ind0 = abs.(mod.cachm.u) .< 1e-15
        any(ind0) ? mod.cachm.u[ind0] .= (sign.(mod.cachm.u[ind0]) .* 0.0) : nothing

        mg = maximum(abs.(mod.cachm.u))
        mg > maxintfloat() && isfinite(mg) ? mod.cachm.u .*= 0.3 : mod.cachm.u[isnan.(mod.cachm.u)] .= 1.0

        return mod.cachm.u
    end

    function d2logp(mod::squiggle, x::Vector{T}) where T <: Number

		mod.cachm.U .= 0.0

        ~isfinite(x[1]) ? x[1] = sign(x[1]) * maxintfloat() : nothing
        ~isfinite(x[2]) ? x[2] = sign(x[2]) * maxintfloat() : nothing
        
        y = zeros(D)
        y[1] = x[1]
        y[2] = x[2] + sin(mod.a * x[1])

        gy = gradlogpdf(mod.MM, y)
		Hy = inv(mod.MM.Σ)
		Jx = [1.0 0.0; mod.a * cos(mod.a * x[1]) 1.0]

		mod.cachm.U -= Jx' * Hy * Jx 
        mod.cachm.U[1, 1] += gy[2] * (-mod.a^2.0 * sin(mod.a * x[1]))

        ind0 = abs.(mod.cachm.U) .< 1e-15 
        any(ind0) ? mod.cachm.U[ind0] .= (sign.(mod.cachm.U[ind0]) .* 0.0) : nothing

        mH = maximum(abs.(mod.cachm.U))
        mH > maxintfloat() && isfinite(mH) ? mod.cachm.U .*= 0.3 : mod.cachm.U[isnan.(mod.cachm.U)] .= 1.0

        return mod.cachm.U
    end

    function G(mod::squiggle, x::Vector{T}, α::T = 1.0) where T <: Number
		# metric-tensor embedding

		g = mod.g(mod, x)
		broadcast!(*, mod.cachm.U, g, g');
		mod.cachm.U .*= α^2.0
		mod.cachm.U[mod.ind] .+= 1.0

		return mod.cachm.U
    end

    function iG(mod::squiggle, x::Vector{T}, α::T = 1.0) where T <: Number
        # inverse metric tensor

		g = mod.g(mod, x)
		L = 1.0 + α^2.0 * norm(g)^2.0
		broadcast!(*, mod.cachm.U, g, g');
		mod.cachm.U .*= - α^2.0 / L
		mod.cachm.U[mod.ind] .+= 1.0

		return mod.cachm.U
    end

	return squiggle(2, a, logp, dlogp, d2logp, G, iG, dist, marg, 
    collect(1:(D+1):D^2),
	cache(zeros(D), zeros(D, D), zeros(D), zeros(D), zeros(D)))
end