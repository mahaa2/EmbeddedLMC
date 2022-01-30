Base.@kwdef mutable struct hprSparse
	mu::Union{Any, prior} = []
	sd::Union{Any, prior} = []
end

Base.@kwdef mutable struct prSparse <: prior
	D::Int64
	mu::Number
	sd::Number
	nu::Number
	pak::Function
	unpak::Function
	lp::Function
	g::Function
	H::Function
	G::Function
	iG::Function
	ind::Vector{Int64}
	cachm::cache
	prior::Union{hprSparse, Any}
end

function prSparse(D::Int64 = 2, 
	mu::Number = 0.0, 
	sd::Number = 10.0, 
	nu::Number = 2.0,
	muprior::Union{Any, prior} = [],
	sdprior::Union{Any, prior} = [])

	# DESCRIPTION :
	# creates a structure for the Sparse prior distribution
	# see https://en.wikipedia.org/wiki/Generalized_normal_distribution

	if (sd <= 0 || nu <= 0)
		@error "sd or nu must be greater than zero"
	end

	function pak(pr::prSparse)
		w = Float64[]

		if typeof(pr.prior.mu) <: prior
			w = vcat(w, pr.mu)
			w = vcat(w, pr.prior.mu.pak(pr.prior.mu))
		end

		if typeof(pr.prior.sd) <: prior
			logsd = isoftplus(pr.sd)
			w = vcat(w, logsd)
			w = vcat(w, pr.prior.sd.pak(pr.prior.sd))
		end

		return w
	end

	function unpak(pr::prSparse, w::Union{T, Vector{T}}) where T <: Number

		if typeof(pr.prior.mu) <: prior
			pr.mu = w[1]
			typeof(w) <: Array ? w = w[2:end] : w = Float64[]
		end

		if typeof(pr.prior.sd) <: prior
			pr.sd = softplus(w[1])
			typeof(w) <: Array ? w = w[2:end] : w = Float64[]
		end

		return (pr, w)
	end

	function lp(pr::prSparse, x::Union{T, Vector{T}}) where T <: Number
		# DESCRIPTION:
		# log prior for the Sparse distribution

		d = x .- pr.mu
		if any(d .== 0.0)
			@error "function is not differentiable at x = µ"
		end

		z = abs.(d) ./ sd
		lp = sum((log(pr.nu) - log(2 * pr.sd) - loggamma(1.0 / pr.nu)) .- (z .^ pr.nu))

		return lp
	end

	function dlp(pr::prSparse, x::Union{T, Vector{T}}) where T <: Number
		# DESCRIPTION:
		# gradient log prior for the Sparse distribution w.r.t
		# its parameters

		pr.cachm.u .= 0.0

		d = x .- pr.mu
		if any(d .== 0.0)
			@error "function is not differentiable at x = µ"
		end

		pr.cachm.u .+= abs.(d) ./ pr.sd
		pr.cachm.u .^= (pr.nu - 1.0)
		pr.cachm.u .*= (pr.nu / pr.sd)

		ifelse.(x .> pr.mu, pr.cachm.u .*= - 1.0, pr.cachm.u)

		return pr.cachm.u
	end

	function d2lp(pr::prSparse, x::Union{T, Vector{T}}) where T <: Number
		# DESCRIPTION:
		# gradient log prior for the Sparse distribution w.r.t
		# its parameters

		pr.cachm.U .= 0.0

		d = x .- pr.mu
		if any(d .== 0.0)
			@error "function is not differentiable at x = µ"
		end

		pr.cachm.U[pr.ind] .+= abs.(d) ./ pr.sd
		pr.cachm.U[pr.ind] .*= sign.(pr.cachm.U[pr.ind])
		pr.cachm.U[pr.ind] .^= (pr.nu - 2.0)
		pr.cachm.U[pr.ind] .*= - (pr.nu * (pr.nu - 1.0) / pr.sd^2.0)

		return pr.cachm.U
	end

	function G(mod::prSparse, x::Vector{T}) where T <: Number
		# metric-tensor embedding

		g = mod.g(mod, x)
		broadcast!(*, mod.cachm.U, g, g');
		mod.cachm.U[mod.ind] .+= 1.0

		return mod.cachm.U

	end

	function iG(mod::prSparse, x::Vector{T}) where T <: Number
        # inverse metric tensor

		g = mod.g(mod, x)
		L = 1.0 + norm(g)^2.0
		broadcast!(*, mod.cachm.U, g, g');
		mod.cachm.U ./= - L;
		mod.cachm.U[mod.ind] .+= 1.0

		return mod.cachm.U

	end

	# pass the structure
	prSparse(D, mu, sd, nu, pak, unpak, lp, dlp, d2lp, G, iG,
	 collect(1:(D+1):D^2), 
	 cache(zeros(D), zeros(D, D), zeros(D), zeros(D), zeros(D)),	 
	 hprSparse(muprior, sdprior))
end
