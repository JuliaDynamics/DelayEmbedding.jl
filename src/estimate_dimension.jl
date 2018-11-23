using NearestNeighbors, Statistics

export estimate_dimension, stochastic_indicator
#####################################################################################
#                                Estimate Dimension                                 #
#####################################################################################

# Function to increase the distance (p-norm) between two points `(i,j)` of
# the embedded time `s`series, by adding one temporal neighbor
function _increase_distance(δ::T, s::AbstractVector, i::Int, j::Int, γ::Int, τ::Int, p) where {T}
    if p==Inf
        return T( max(δ, abs(s[i+γ*τ+τ] - s[j+γ*τ+τ])) )
    else
        return T( (δ^p + abs(s[i+γ*τ+τ] - s[j+γ*τ+τ])^p )^(1/p) )
    end
end

"""
    estimate_dimension(s::AbstractVector, τ:Int, γs = 1:5) -> E₁s

Compute a quantity that can estimate an optimal amount of
temporal neighbors `γ` to be used in [`reconstruct`](@ref) or [`embed`](@ref).

## Description
Given the scalar timeseries `s` and the embedding delay `τ` compute the
values of `E₁` for each `γ ∈ γs`, according to Cao's Method (eq. 3 of [1]).
Please be aware that in **DynamicalSystems.jl** `γ` stands for the amount of temporal
neighbors and not the embedding dimension (`D = γ + 1`, see also [`embed`](@ref)).

Return the vector of all computed `E₁`s. To estimate a good value for `γ` from this,
find `γ` for which the value `E₁` saturates at some value around 1.

*Note: This method does not work for datasets with perfectly periodic signals.*

## References

[1] : Liangyue Cao, [Physica D, pp. 43-50 (1997)](https://www.sciencedirect.com/science/article/pii/S0167278997001188?via%3Dihub)
"""
estimate_dimension(s,γ,τ=1:5) = afnn(s,γ,τ,Inf) # By default it is `afnn` with Inf-norm

function afnn(s::AbstractVector{T}, τ::Int, γs = 1:5, p=Inf) where {T}
    E1s = zeros(length(γs))
    aafter = 0.0
    aprev = _average_a(s, γs[1], τ, p)
    for (i, γ) ∈ enumerate(γs)
        aafter = _average_a(s, γ+1, τ, p)
        E1s[i] = aafter/aprev
        aprev = aafter
    end
    return E1s
end
# then use function `saturation_point(γs, E1s)` from ChaosTools

function _average_a(s::AbstractVector{T},γ,τ,p) where {T}
    #Sum over all a(i,d) of the Ddim Reconstructed space, equation (2)
    R2 = reconstruct(s[1:end-τ],γ,τ)
    tree2 = KDTree(R2)
    nind = (x = knn(tree2, R2.data, 2)[1]; [ind[1] for ind in x])
    e=0.
    for (i,j) ∈ enumerate(nind)
        δ = norm(R2[i]-R2[j], p)
        #If R2[i] and R2[j] are still identical, choose the next nearest neighbor
        if δ == 0.
            j = knn(tree2, R2[i], 3, true)[1][end]
            δ = norm(R2[i]-R2[j], p)
        end
        e += _increase_distance(δ,s,i,j,γ,τ,p)/δ
    end
    return e / (length(R2)-1)
end

function dimension_indicator(s,γ,τ,p=Inf) #this is E1, equation (3) of Cao
    return _average_a(s,γ+1,τ,p)/_average_a(s,γ,τ,p)
end


"""
    stochastic_indicator(s::AbstractVector, τ:Int, γs = 1:4) -> E₂s

Compute an estimator for apparent randomness in a reconstruction with `γs` temporal
neighbors.

## Description
Given the scalar timeseries `s` and the embedding delay `τ` compute the
values of `E₂` for each `γ ∈ γs`, according to Cao's Method (eq. 5 of [1]).

Use this function to confirm that the
input signal is not random and validate the results of [`estimate_dimension`](@ref).
In the case of random signals, it should be `E₂ ≈ 1 ∀ γ`.

## References

[1] : Liangyue Cao, [Physica D, pp. 43-50 (1997)](https://www.sciencedirect.com/science/article/pii/S0167278997001188?via%3Dihub)
"""
function stochastic_indicator(s::AbstractVector{T},τ, γs=1:4) where T # E2, equation (5)
    #This function tries to tell the difference between deterministic
    #and stochastic signals
    #Calculate E* for Dimension γ+1
    E2s = Float64[]
    for γ ∈ γs
        R1 = reconstruct(s,γ+1,τ)
        tree1 = KDTree(R1[1:end-1-τ])
        method = FixedMassNeighborhood(2)

        Es1 = 0.
        nind = (x = neighborhood(R1[1:end-τ], tree1, method); [ind[1] for ind in x])
        for  (i,j) ∈ enumerate(nind)
            Es1 += abs(R1[i+τ][end] - R1[j+τ][end]) / length(R1)
        end

        #Calculate E* for Dimension γ
        R2 = reconstruct(s,γ,τ)
        tree2 = KDTree(R2[1:end-1-τ])
        Es2 = 0.
        nind = (x = neighborhood(R2[1:end-τ], tree2, method); [ind[1] for ind in x])
        for  (i,j) ∈ enumerate(nind)
            Es2 += abs(R2[i+τ][end] - R2[j+τ][end]) / length(R2)
        end
        push!(E2s, Es1/Es2)
    end
    return E2s
end

"""
    fnn(s::AbstractVector, τ:Int, γs = 1:5, Rtol=10., Atol=2.)

Calculate the number of "false nearest neighbors" (FNN) of the datasets created
from `s` with a sequence of `τ`-delayed temporal neighbors.

## Description
Given a dataset made by embedding `s` with `γ` temporal neighbors and delay `τ`,
the "false nearest neighbors" (FNN) are the pairs of points that are nearest to
each other at dimension `γ`, but are separated at dimension `γ+1`. Kennel's
criteria for detecting FNN are based on a threshold for the relative increment
of the distance between the nearest neighbors (`Rtol`, eq. 4 in [1]), and
another threshold for the ratio between the increased distance and the
"size of the attractor" (`Atol`, eq. 5 in [1]).

The returned value is a vector with the number of FNN for each `γ ∈ γs`. The
optimal value for `γ` is found at the point where the number of FNN approaches
zero.

Please be aware that in **DynamicalSystems.jl** `γ` stands for the amount of temporal
neighbors and not the embedding dimension (`D = γ + 1`, see also [`embed`](@ref)).

See also: [`estimate_dimension](@ref), [`f1nn`](@ref).

## References

[1] : M. Kennel *et al.*, "Determining embedding dimension for phase-space
reconstruction using a geometrical construction", *Phys. Review A 45*(6), 3403-3411
(1992).
"""
function fnn(s::AbstractVector, τ::Int, γs = 1:5, Rtol=10., Atol=2.)
    Rtol2 = Rtol^2
    Ra = std(s, corrected=false)
    nfnn = zeros(length(γs))
    for (k, γ) ∈ enumerate(γs)
        y = reconstruct(s[1:end-τ],γ,τ)
        tree = KDTree(y)
        nind = (x = knn(tree, y.data, 2)[1]; [ind[1] for ind in x])
        for (i,j) ∈ enumerate(nind)
            δ = norm(y[i]-y[j], 2)
            # If y[i] and y[j] are still identical, choose the next nearest neighbor
            # as in Cao's algorithm (not suggested by Kennel, but still advisable)
            if δ == 0.
                j = knn(tree, y[i], 3, true)[1][end]
                δ = norm(y[i]-y[j], 2)
            end
            δ1 = _increase_distance(δ,s,i,j,γ,τ,2)
            cond_1 = ((δ1/δ)^2 - 1 > Rtol2) # equation (4) of Kennel
            cond_2 = (δ1/Ra > Atol)         # equation (5) of Kennel
            if cond_1 | cond_2
                nfnn[k] += 1
            end
        end
    end
    return nfnn
end

"""
    f1nn(s::AbstractVector, τ:Int, γs = 1:5, Rtol=10., Atol=2.)

Calculate the ratio of "false first nearest neighbors" (FFNN) of the datasets created
from `s` with a sequence of `τ`-delayed temporal neighbors.

## Description
Given a dataset made by embedding `s` with `γ` temporal neighbors and delay `τ`,
the "first nearest neighbors" (FFNN) are the pairs of points that are nearest to
each other at dimension `γ` that cease to be nearest neighbors at dimension
`γ+1` [1].

The returned value is a vector with the ratio between the number of FFNN and
the number of points in the dataset for each `γ ∈ γs`. The optimal value for `γ`
is found at the point where this ratio approaches zero.

Please be aware that in **DynamicalSystems.jl** `γ` stands for the amount of temporal
neighbors and not the embedding dimension (`D = γ + 1`, see also [`embed`](@ref)).

See also: [`estimate_dimension](@ref), [`fnn`](@ref).

## References

[1] : Anna Krakovská *et al.*, "Use of false nearest neighbours for selecting
variables and embedding parameters for state space reconstruction", *J Complex
Sys* 932750 (2015), DOI: 10.1155/2015/932750
"""
function f1nn(s::AbstractVector, τ::Int, γs = 1:5)
    f1nn_ratio = zeros(length(γs))
    γ_prev = 0 # to recall what γ has been analyzed before
    Rγ = reconstruct(s[1:end-τ],γs[1],τ) # this is for the first iteration 
    for (i, γ) ∈ enumerate(γs)
        if i>1 && γ!=γ_prev+1
            # Re-calculate the series with γ delayed dims if γ does not follow
            # the dimension of the previous iteration 
            Rγ = reconstruct(s[1:end-τ],γ,τ)
        end
        (nf1nn, Rγ) = _compare_first_nn(s,γ,τ,Rγ)
        f1nn_ratio[i] = nf1nn/length(Rγ)
        # Trim Rγ for the next iteration
        Rγ = Rγ[1:end-τ,:]
        γ_prev = γ
    end
    return f1nn_ratio
end

function _compare_first_nn(s::AbstractVector{T},γ::Int,τ::Int,Rγ::Dataset{D,T}) where {D} where {T}
    # This function compares the first nearest neighbors of `s`
    # embedded with Dimensions `γ` and `γ+1` (the former given as input)
    tree = KDTree(Rγ)
    Rγ1 = reconstruct(s,γ+1,τ)
    tree1 = KDTree(Rγ1)
    nf1nn = 0
    # For each point `i`, the fnn of `Rγ` is `j`, and the fnn of `Rγ1` is `k` 
    nind = (x = knn(tree, Rγ.data, 2)[1]; [ind[1] for ind in x])
    for  (i,j) ∈ enumerate(nind)
        k = knn(tree1, Rγ1.data[i], 2, true)[1][end]
        if j != k
            nf1nn += 1
        end
    end
    # `R1` is returned to re-use it if necessary
    return (nf1nn, Rγ1)
end
