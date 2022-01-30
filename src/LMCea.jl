function LMCea(MM::Union{targdist, prior} = M, 
    x::Vector{Float64} = randn(2),
    nsamp::Int64 = 100, 
    ε2::Float64 = 0.01,
    LF::Int64 = 10,
    adapt::String = "0",
    α::Float64 = 1.0,
    vini::Union{Any, Vector{Float64}} = [])

    # DESCRIPTION : This program takes as arguments 
    #     a model structure (M),
    #     an initial position vector (x),
    #     mcmc sample-size (nsamp),
    #     step-size (ε2),
    #     number of leapfrog steps (LF),
    #     step-size adaptation (adatp, has been used for test purposes only)
    #     tempered target distribution (α)
    #     initial velocity vector if given (vini) and
    # 
    # return a matrix representing the samples from 
    # a target distribution structured in M.

    X  = zeros(MM.D, nsamp)         # matrix for the targets distribution samples
    xn = copy(x)                    # initial position
    vn = zeros(MM.D)                # initial velocity
    xnI = zeros(MM.D)               # allocate mem for initial fixed position
    ∇l = zeros(MM.D)                # allocate mem for the gradient 
    srootG = zeros(MM.D, MM.D)      # allocate mem for square-root metric tensor
    accr = 0.0                      # acceptance rate 

    idx = 1:nsamp
    idl = 1:LF
    
    # looping for all the samples
    @inbounds @fastmath @simd for ns in eachindex(idx)
        xnI = copy(xn)
        ∇l .+= MM.g(MM, xnI)
        nL = norm(∇l)^2.0
        α2 = α^2.0
        L = 1.0 + α2 * nL
        
        # Let's work with the "L-normalised" gradient
        ∇l ./= sqrt(L) 

        if ns == 1 & ~isempty(vini)
            vn .+= vini 

        else
            if α <= eps()
                srootG .= 0.0
                srootG[MM.ind] .+= 1.0

            else
                nL > 1e-10 ? c = (L / nL) * (1.0 / sqrt(L) - 1.0) : c = -0.5 * α2
                broadcast!(*, srootG, ∇l, ∇l')
                srootG .*= c
                srootG[MM.ind] .+= 1.0

            end

            # sample initial velocity and overwrite
            LinearAlgebra.BLAS.symv!('L', 1.0, srootG, randn(MM.D), 0.0, vn)
        end

        # initial energy
        Eini = ( - MM.lp(MM, xnI) - 0.5 * log(L) + 
            0.5 * norm(vn)^2 + 0.5 * L * α2 * dot(vn, ∇l)^2.0 )

        ∇l .= 0.0

        # purposes test only 
        if isequal(adapt, "0")
            ε = ε2 * 1.0

        elseif isequal(adapt, "1")
            ε = ε2 * sqrt(L)

        elseif isequal(adapt, "2")
            e = diag(MM.H(MM, xnI))
            ε = ε2 * sqrt(L) / maximum(e)

        elseif isequal(adapt, "3")
            ε = ε2 / dot(vn ./ norm(vn), MM.H(MM, xnI) ./ sqrt(L), vn ./ norm(vn))

        end

        randn() > 0.0 ? ε *= 1.0 : ε *= -1.0

        # initialize Δ log det 
        Δlogdet = [0.0]
    
        # numeric-geometric integrator loop
        for n in eachindex(idl)
            # update velocity explicitly with half-step and Δ log det 
            lpfroga!(MM, xn, vn, Δlogdet, ε, α)

            # update position with full step
            LinearAlgebra.BLAS.axpy!(ε, vn, xn)

            # update velocity explicitly with full-step and Δ log det 
            lpfroga!(MM, xn, vn, Δlogdet, ε, α)

        end

        # Calculate the proposed energy function
        ∇l .+= MM.g(MM, xn)
        L = 1.0 + α2 * norm(∇l)^2.0
        ∇l ./= sqrt(L) 
        
        Enew = (- MM.lp(MM, xn) - 0.5 * log(L) + 
            0.5 * norm(vn)^2 + 0.5 * L * α2 * dot(vn, ∇l)^2.0 )
        
        ∇l .= 0.0

        # accept according to metropolis-hastings
        logratio = - Enew + Eini + Δlogdet[]

        if isfinite(logratio) && logratio > minimum([0.0, log(rand())])
            X[:, ns] .+= xn
            accr += 1.0
        else
            X[:, ns] .+= xnI
            xn = copy(xnI)
        end
    end
    display(accr / nsamp)

    return X
end

function lpfroga!(MM::Union{targdist, prior}, 
    x::Vector{Float64},
    v::Vector{Float64},
    Δlogdet::Vector{Float64},
    ε::Float64,
    α::Float64) 

    # DESCRIPTION : This programs takes as arguments
    #      a model structure (M),
    #      a initial position (x),
    #      a initial vecocity (v),
    #      elements of the Jacobian transformation of R.V (Δlogdet),
    #      step-size (ε),
    #      temperered distribution (α), 
    # 
    # and overwrites the values of v and Δlogdet

    # "L-normalized" gradient
    ∇l   = MM.g(MM, x)
    α2   = α^2.0
    L    = 1.0 + α2 * norm(∇l)^2.0
    sL   = sqrt(L)
    ∇l ./= sL

    # Hessian (should we take care of STABILITY in here ?)
    α > eps() ? H = MM.H(MM, x) : H = zeros(MM.D, MM.D)
    H  ./= sL

    # starting calculate updates
    LinearAlgebra.BLAS.symv!('L', 1.0, H, ∇l, 0.0, MM.cachm.∇lTH)
    LinearAlgebra.BLAS.symv!('L', 1.0, H, v,  0.0, MM.cachm.vTH)
    MM.cachm.vTH .*= (ε / 2.0)
    L3 = dot(∇l, MM.cachm.vTH)
    L4 = dot(MM.cachm.vTH, v)
    L5 = dot(v, ∇l) 
    L1 = 1.0 / (norm(∇l).^2.0 + L3 + 1.0 / (L * α2))

    # update log det with position and velocity
    Δlogdet .-= log(abs(1.0 + α2 * L3))

    # overwriting the updates for memory allocations
    H .*= - α2 * (ε / 2.0)
    H[MM.ind] .+= (ε * sL / 2.0) + α2 * L * L5
    MM.cachm.vTH .+= ∇l
    LinearAlgebra.BLAS.symv!('L', 1.0, H, MM.cachm.vTH, 0.0, MM.cachm.Hv)
    LinearAlgebra.BLAS.gemm!('N', 'T', - L1, ∇l, MM.cachm.Hv, 1.0, H)

    L1 *= (L5 + L4)
    H[MM.ind] .-= L1

    # 1/2-update velocity
    LinearAlgebra.BLAS.gemm!('N', 'N', 1.0, H, ∇l, 1.0, v)

    # update log det with position and 1/2-update velocity
    Δlogdet .+= log(abs(1.0 - (ε * α2 / 2.0) * dot(MM.cachm.∇lTH, v)))
end