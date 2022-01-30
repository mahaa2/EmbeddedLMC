Base.@kwdef mutable struct bansh <: targdist
    D::Union{Int64, Any} = 2
    sd::Float64 = 2.0
	st::Float64 = 1.0
	y::Union{Vector{Float64}, Float64}
    lp::Union{Function, Any} = []
    g::Union{Function, Any} = []
    H::Union{Function, Any} = []
	G::Union{Function, Any} = []
	iG::Union{Function, Any} = []
    ind::Vector{Int64} = collect(1:3:4)
	cachm::cache
	F::Union{Function, Any} = []
end

function bansh(y::Union{Vector{Float64}, Float64},
	           sd::Float64 = 2.0,
			   st::Float64 = 1.0, D::Int64 = 2)

    function logp(mod::bansh, x::Vector{T}) where T <: Number
		mu = x[1] + x[2]^2.0
		
        return sum(logpdf.(Normal(mu, mod.sd), mod.y)) +
		sum(logpdf.(Normal(0.0, mod.st), x))
    end

    function dlogp(mod::bansh, x::Vector{T}) where T <: Number

		mod.cachm.u .= 0.0

		mu = x[1] + x[2]^2.0

		mod.cachm.u[1] += sum((mod.y .- mu) ./ mod.sd^2.0) - x[1] / mod.st^2.0
		mod.cachm.u[2] += sum((mod.y .- mu) ./ mod.sd^2.0 .* 2.0 .* x[2]) - x[2] / mod.st^2.0

		ind0 = abs.(mod.cachm.u) .< 1e-15
        any(ind0) ? mod.cachm.u[ind0] .= (sign.(mod.cachm.u[ind0]) .* 0.0) : nothing

        mg = maximum(abs.(mod.cachm.u))
        mg > maxintfloat() && isfinite(mg) ? mod.cachm.u .*= 0.3 : mod.cachm.u[isnan.(mod.cachm.u)] .= 1.0

        return mod.cachm.u
    end

    function d2logp(mod::bansh, x::Vector{T}) where T <: Number

		mod.cachm.U .= 0.0

		mu = x[1] + x[2]^2.0
		ny = length(y)

        mod.cachm.U[1, 1] += - ny / mod.sd^2.0 - 1.0 / mod.st^2.0
        mod.cachm.U[1, 2] = mod.cachm.U[2, 1] += - ny / mod.sd^2.0 * 2.0 * x[2]
        mod.cachm.U[2, 2] += - (ny / mod.sd^2.0 * 4.0 * x[2]^2.0 +
		 			  sum(.- (mod.y .- mu) ./ mod.sd^2.0 .* 2.0)) - 1.0 / mod.st^2.0

        ind0 = abs.(mod.cachm.U) .< 1e-15 
        any(ind0) ? mod.cachm.U[ind0] .= (sign.(mod.cachm.U[ind0]) .* 0.0) : nothing

        mH = maximum(abs.(mod.cachm.U))
        mH > maxintfloat() && isfinite(mH) ? mod.cachm.U .*= 0.3 : mod.cachm.U[isnan.(mod.cachm.U)] .= 1.0

        return mod.cachm.U
    end

    function G(mod::bansh, x::Vector{T}, α::T = 1.0) where T <: Number
		# metric-tensor embedding

		g = mod.g(mod, x)
		broadcast!(*, mod.cachm.U, g, g');
		mod.cachm.U .*= α^2.0
		mod.cachm.U[mod.ind] .+= 1.0

		return mod.cachm.U
    end

    function iG(mod::bansh, x::Vector{T}, α::T = 1.0) where T <: Number
        # inverse metric tensor

		g = mod.g(mod, x)
		L = 1.0 + α^2.0 * norm(g)^2.0
		broadcast!(*, mod.cachm.U, g, g');
		mod.cachm.U .*= - α^2.0 / L
		mod.cachm.U[mod.ind] .+= 1.0

		return mod.cachm.U
    end

	function F(mod::bansh, x::Vector{T}) where T <: Number
		# Fisher information matrix

		mod.cachm.U .= 0.0

		ny = length(y)

		mod.cachm.U[1, 1] += ny / mod.sd^2.0 + 1.0 / mod.st^2.0
		mod.cachm.U[1, 2] = mod.cachm.U[2, 1] += 2.0 * ny * x[2] / mod.sd^2.0
		mod.cachm.U[2, 2] += 4.0 * ny * x[2]^2.0 / mod.sd^2.0 + 1.0 / mod.st^2.0


		ind0 = abs.(mod.cachm.U) .< 1e-15 
        any(ind0) ? mod.cachm.U[ind0] .= (sign.(mod.cachm.U[ind0]) .* 0.0) : nothing

        mH = maximum(abs.(mod.cachm.U))
        mH > 1e5 && isfinite(mH) ? mod.cachm.U ./= (mH / 2.0) : nothing

		return mod.cachm.U
    end

	return bansh(D, sd, st, y, logp, dlogp, d2logp, G, iG, collect(1:(D+1):D^2),
	cache(zeros(D), zeros(D, D), zeros(D), zeros(D), zeros(D)), F)
end
