Base.@kwdef mutable struct hprlogreg
	th::Union{Any, prior} = []
end

Base.@kwdef mutable struct logreg <: targdist
	D::Int64 = 2
	pak::Union{Function, Any} = []
	unpak::Union{Function, Any} = []
	lp::Union{Function, Any} = []
	g::Union{Function, Any} = []
	H::Union{Function, Any} = []
	G::Union{Function, Any} = []
	iG::Union{Function, Any} = []
	ind::Vector{Int64} = []
	cachm::cache
	y::Vector{Float64} = zeros(2)
	A::Matrix{Float64} = zeros(2, 2)
	prior::Union{hprlogreg, Any} = []
end

function logreg(D::Int64 = 2, thprior::Union{Any, prior} = [])

	# DESCRIPTION :
	# creates a likelihood-structure for a
	# linear regression model with Gaussian errors

	function pak(lik::logreg)
		w = Float64[]

		if typeof(lik.prior.sd) <: prior
			logsd = isoftplus(lik.sd)
			w = vcat(w, logsd)
			w = vcat(w, lik.prior.sd.pak(lik.prior.sd))
		end

		return w
	end

	function unpak(lik::logreg, w::Union{T, Vector{T}}) where T <: Number
        
		if typeof(lik.prior.sd) <: prior
			lik.sd =  softplus(w[1])
			typeof(w) <: Array ? w = w[2:end] : w = Float64[]
		end

		return (lik, w)
	end

	function lp(lik::logreg, x::Union{T, Vector{T}}) where T <: Number
		# DESCRIPTION:
		# likelihood structure for a linear regression model

		µ = lik.A * x
		lp = µ' * lik.y - sum(log.(1.0 .+ exp.(µ)))

		if typeof(lik.prior.th) <: prior
			lp += lik.prior.th.lp(lik.prior.th, x)
		end

		return lp
	end

	function dlp(lik::logreg, x::Union{T, Vector{T}}) where T <: Number
		# DESCRIPTION:
		# gradient log likelihood w.r.t x

		lik.cachm.u .= 0.0

		µ = lik.A * x
		lik.cachm.u .+= lik.A' * (lik.y .- 1.0 ./ (1.0 .+ exp.(-µ)))

		if typeof(lik.prior.th) <: prior
			lik.cachm.u .+= lik.prior.th.g(lik.prior.th, x)
		end

        ind0 = abs.(lik.cachm.u) .< eps()
        any(ind0) ? lik.cachm.u[ind0] .= (sign.(lik.cachm.u[ind0]) .* 0.0) : nothing

        mg = maximum(abs.(lik.cachm.u))
        # mg > 1e6 && isfinite(mg) ? mod.cachm.u ./= (0.8 * mg) : nothing
        mg > maxintfloat() && isfinite(mg) ? lik.cachm.u .*= 0.3 : lik.cachm.u[isnan.(lik.cachm.u)] .= 0.0

		return lik.cachm.u
	end

	function d2lp(lik::logreg, x::Union{T, Vector{T}}) where T <: Number
		# DESCRIPTION:
		# hessian

		lik.cachm.U .= 0.0

		p = 1.0 ./ (1.0 .+ exp.(- lik.A * x))
		s = sqrt.((1.0 .- p) .* p)
		S = broadcast(*, s, lik.A)

		LinearAlgebra.BLAS.gemm!('T', 'N', -1.0, S, S, 1.0, lik.cachm.U)
		lik.cachm.U .+= lik.cachm.U'
		lik.cachm.U ./= 2.0

		if typeof(lik.prior.th) <: prior
			lik.cachm.U .+= lik.prior.th.H(lik.prior.th, x)
		end

		ind0 = abs.(lik.cachm.U) .< eps()
		any(ind0) ? lik.cachm.U[ind0] .= sign.(lik.cachm.U[ind0]) .* 0.0 : nothing

        mH = maximum(abs.(lik.cachm.U))
        # mH > 1e15 && isfinite(mH) ? mod.cachm.U ./= (0.8 * mH) : nothing
        mH > maxintfloat() && isfinite(mH) ? lik.cachm.U .*= 0.3 : lik.cachm.U[isnan.(lik.cachm.U)] .= 0.0

		return lik.cachm.U
	end

	function G(lik::logreg, x::Vector{T}) where T <: Number
		# metric-tensor embedding

		g = lik.g(lik, x)
		broadcast!(*, lik.cachm.U, g, g');
		lik.cachm.U[lik.ind] .+= 1.0

		return lik.cachm.U

	end

	function iG(mod::logreg, x::Vector{T}) where T <: Number
		# inverse metric-tensor
	
		g = lik.g(lik, x)
		L = 1.0 + norm(g)^2.0
		broadcast!(*, lik.cachm.U, g, g');
		lik.cachm.U ./= - L;
		lik.cachm.U[mod.ind] .+= 1.0

		return lik.cachm.U
	end

	# pass the structure
	logreg(D, pak, unpak, lp, dlp, d2lp, G, iG, collect(1:(D + 1):D^2),
	cache(zeros(D), zeros(D, D), zeros(D), zeros(D), zeros(D)),
	zeros(5), zeros(5, D), hprlogreg(thprior))
end