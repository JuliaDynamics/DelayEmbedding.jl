using StaticArrays, LinearAlgebra
using Base.Iterators: flatten

export Dataset, AbstractDataset, SVector, minima, maxima
export minmaxima, columns, regularize, dimension

abstract type AbstractDataset{D, T} end

"""
    dimension(thing) -> D
Return the dimension of the `thing`, in the sense of state-space dimensionality.
"""
dimension(::AbstractDataset{D,T}) where {D,T} = D
@inline Base.eltype(d::AbstractDataset{D,T}) where {D,T} = T
import Base: ==
==(d1::AbstractDataset, d2::AbstractDataset) = d1.data == d2.data

# Size:
@inline Base.length(d::AbstractDataset) = length(d.data)
@inline Base.size(d::AbstractDataset{D,T}) where {D,T} = (length(d.data), D)
@inline Base.size(d::AbstractDataset, i) = size(d)[i]
@inline Base.IteratorSize(d::AbstractDataset) = Base.HasLength()

# Iteration interface:
@inline Base.eachindex(D::AbstractDataset) = Base.OneTo(length(D.data))
@inline Base.iterate(d::AbstractDataset, state = 1) = iterate(d.data, state)
@inline Base.eltype(::Type{<:AbstractDataset{D, T}}) where {D, T} = SVector{D, T}
Base.eachcol(ds::AbstractDataset) = (ds[:, i] for i in 1:size(ds, 2))
Base.eachrow(ds::AbstractDataset) = ds.data

# 1D indexing over the container elements:
@inline Base.getindex(d::AbstractDataset, i::Int) = d.data[i]
@inline Base.getindex(d::AbstractDataset, i) = Dataset(d.data[i])
@inline Base.lastindex(d::AbstractDataset) = length(d)
@inline Base.lastindex(d::AbstractDataset, k) = size(d)[k]
@inline Base.firstindex(d::AbstractDataset) = 1
@inline Base.setindex!(d::AbstractDataset, v, i::Int) = (d.data[i] = v)

# 2D indexing exactly like if the dataset was a matrix
# with each column a dynamic variable
@inline Base.getindex(d::AbstractDataset, i::Int, j::Int) = d.data[i][j]
@inline Base.getindex(d::AbstractDataset, ::Colon, j::Int) =
[d.data[k][j] for k in 1:length(d)]
@inline Base.getindex(d::AbstractDataset, i::Int, ::Colon) = d.data[i]
# Indexing with ranges
@inline Base.getindex(d::AbstractDataset, i::AbstractVector, j::Int) =
[d.data[k][j] for k in i]
@inline Base.getindex(d::AbstractDataset, i::AbstractVector, ::Colon) =
Dataset([d[k] for k in i])
@inline Base.getindex(d::AbstractDataset, i::Int, j::AbstractVector) =
[d.data[i][k] for k in j]
@inline Base.getindex(d::AbstractDataset, ::Colon, ::Colon) = d
# Indexing with boolean vectors
Base.getindex(d::AbstractDataset, i::AbstractVector{Bool}, ::Colon) =
Dataset(d[i])

# this function could be done generated
function Base.getindex(d::AbstractDataset{D,T}, i::AbstractVector{Int},
    j::AbstractVector{Int}) where {D,T}
    I = length(i)
    J = length(j)
    ret = zeros(T, J, I)
    for k=1:I
        for l=1:J
            ret[l,k] = d[i[k],j[l]]
        end
    end
    return Dataset(transpose(ret))
end

function Base.getindex(d::AbstractDataset{D,T},
    ::Colon, j::AbstractVector{Int}) where {D, T}
    return Dataset(Base.getindex(d, 1:length(d), j))
end

"""
    columns(dataset) -> x, y, z, ...
Return the individual columns of the dataset.
"""
function columns end
@generated function columns(data::AbstractDataset{D, T}) where {D, T}
    gens = [:(data[:, $k]) for k=1:D]
    quote tuple($(gens...)) end
end

###########################################################################
# appending data
###########################################################################
Base.append!(d1::AbstractDataset, d2::AbstractDataset) = (append!(d1.data, d2.data); d1)
Base.push!(d::AbstractDataset, new_item) = (push!(d.data, new_item); d)

function Base.hcat(d::AbstractDataset{D, T}, x::Vector{<:Real}) where {D, T}
    L = length(d)
    L == length(x) || error("dataset and vector must be of same length")
    data = Vector{SVector{D+1, T}}(undef, L)
    @inbounds for i in 1:L
        data[i] = SVector{D+1, T}(d[i]..., x[i])
    end
    return Dataset(data)
end

function Base.hcat(x::Vector{<:Real}, d::AbstractDataset{D, T}) where {D, T}
    L = length(d)
    L == length(x) || error("dataset and vector must be of same length")
    data = Vector{SVector{D+1, T}}(undef, L)
    @inbounds for i in 1:L
        data[i] = SVector{D+1, T}(x[i], d[i]...)
    end
    return Dataset(data)
end

function Base.hcat(x::AbstractDataset{D1, T}, y::AbstractDataset{D2, T}) where {D1, D2, T}
    length(x) == length(y) || error("Datasets must be of same length")
    L = length(x)
    D = D1 + D2
    v = Vector{SVector{D, T}}(undef, L)
    for i = 1:L
        v[i] = SVector{D, T}((x[i]..., y[i]...,))
    end
    return Dataset(v)
end

###########################################################################
# Concrete implementation
###########################################################################
"""
    Dataset{D, T} <: AbstractDataset{D,T}
A dedicated interface for datasets.
It contains *equally-sized datapoints* of length `D`, represented by `SVector{D, T}`.
These data are contained in the field `.data` of a dataset, as a standard Julia
`Vector{SVector}`.

When indexed with 1 index, a `dataset` is like a vector of datapoints.
When indexed with 2 indices it behaves like a matrix that has each of the columns be the
timeseries of each of the variables.

`Dataset` also supports most sensible operations like `append!, push!, hcat, eachrow`,
among others, and when iterated over, it iterates over its contained points.

## Description of indexing
In the following let `i, j` be integers,  `typeof(data) <: AbstractDataset`
and `v1, v2` be `<: AbstractVector{Int}` (`v1, v2` could also be ranges).

* `data[i]` gives the `i`th datapoint (returns an `SVector`)
* `data[v1] == data[v1, :]`, returns a `Dataset` with the points in those indices.
* `data[:, j]` gives the `j`th variable timeseries, as `Vector`
* `data[v1, v2]` returns a `Dataset` with the appropriate entries (first indices
  being "time"/point index, while second being variables)
* `data[i, j]` value of the `j`th variable, at the `i`th timepoint

Use `Matrix(dataset)` or `Dataset(matrix)` to convert. It is assumed
that each *column* of the `matrix` is one variable.
If you have various timeseries vectors `x, y, z, ...` pass them like
`Dataset(x, y, z, ...)`. You can use `columns(dataset)` to obtain the reverse,
i.e. all columns of the dataset in a tuple.
"""
struct Dataset{D, T} <: AbstractDataset{D,T}
    data::Vector{SVector{D,T}}
end
# Empty dataset:
Dataset{D, T}() where {D,T} = Dataset(SVector{D,T}[])

# Identity constructor:
Dataset{D, T}(s::Dataset{D, T}) where {D,T} = s
Dataset(s::Dataset) = s

###########################################################################
# Dataset(Vectors of stuff)
###########################################################################
Dataset(s::AbstractVector{T}) where {T} = Dataset(SVector.(s))

function Dataset(v::Vector{<:AbstractArray{T}}) where {T<:Number}
    D = length(v[1])
    D > 100 && @warn "You are attempting to make a Dataset of dimensions > 100"
    L = length(v)
    data = Vector{SVector{D, T}}(undef, L)
    for i in 1:length(v)
        D != length(v[i]) && throw(ArgumentError(
        "All data-points in a Dataset must have same size"
        ))
        @inbounds data[i] = SVector{D,T}(v[i])
    end
    return Dataset{D, T}(data)
end

@generated function _dataset(vecs::Vararg{<:AbstractVector{T},D}) where {D, T}
    gens = [:(vecs[$k][i]) for k=1:D]
    D > 100 && @warn "You are attempting to make a Dataset of dimensions > 100"
    quote
        L = minimum(length(x) for x in vecs)
        data = Vector{SVector{$D, T}}(undef, L)
        for i in 1:L
            @inbounds data[i] = SVector{$D, T}($(gens...))
        end
        data
    end
end

function Dataset(vecs::Vararg{<:AbstractVector{T}}) where {T}
    return Dataset(_dataset(vecs...))
end

Dataset(x::AbstractDataset{D1, T}, y::AbstractDataset{D2, T}) where {D1, D2, T} =
    hcat(x, y)
Dataset(x::Vector{<:Real}, y::AbstractDataset{D, T}) where {D, T} = hcat(x, y)
Dataset(x::AbstractDataset{D, T}, y::Vector{<:Real}) where {D, T} = hcat(x, y)

#####################################################################################
#                                Dataset <-> Matrix                                 #
#####################################################################################
function Base.Matrix{S}(d::AbstractDataset{D,T}) where {S, D, T}
    mat = Matrix{S}(undef, length(d), D)
    for j in 1:D
        for i in 1:length(d)
            @inbounds mat[i,j] = d.data[i][j]
        end
    end
    mat
end
Base.Matrix(d::AbstractDataset{D,T}) where {D, T} = Matrix{T}(d)

function Dataset(mat::AbstractMatrix{T}) where {T}
    N, D = size(mat)
    D > 100 && @warn "You are attempting to make a Dataset of dimensions > 100"
    D > N && @warn "You are attempting to make a Dataset of a matrix with more columns than rows."
    Dataset{D,T}(reshape(reinterpret(SVector{D,T}, vec(transpose(mat))), (N,)))
end

#####################################################################################
#                                   Pretty Printing                                 #
#####################################################################################
function Base.summary(d::Dataset{D, T}) where {D, T}
    N = length(d)
    return "$D-dimensional Dataset{$(T)} with $N points"
end

function matstring(d::AbstractDataset{D, T}) where {D, T}
    N = length(d)
    if N > 50
        mat = zeros(eltype(d), 50, D)
        for (i, a) in enumerate(flatten((1:25, N-24:N)))
            mat[i, :] .= d[a]
        end
    else
        mat = Matrix(d)
    end
    s = sprint(io -> show(IOContext(io, :limit=>true), MIME"text/plain"(), mat))
    s = join(split(s, '\n')[2:end], '\n')
    tos = summary(d)*"\n"*s
    return tos
end

Base.show(io::IO, ::MIME"text/plain", d::AbstractDataset) = print(io, matstring(d))
Base.show(io::IO, d::AbstractDataset) = print(io, summary(d))

#####################################################################################
#                                 Minima and Maxima                                 #
#####################################################################################
"""
    minima(dataset)
Return an `SVector` that contains the minimum elements of each timeseries of the
dataset.
"""
function minima(data::AbstractDataset{D, T}) where {D, T<:Real}
    m = Vector(data[1])
    for point in data
        for i in 1:D
            if point[i] < m[i]
                m[i] = point[i]
            end
        end
    end
    return SVector{D,T}(m)
end

"""
    maxima(dataset)
Return an `SVector` that contains the maximum elements of each timeseries of the
dataset.
"""
function maxima(data::AbstractDataset{D, T}) where {D, T<:Real}
    m = Vector(data[1])
    for point in data
        for i in 1:D
            if point[i] > m[i]
                m[i] = point[i]
            end
        end
    end
    return SVector{D, T}(m)
end

"""
    minmaxima(dataset)
Return `minima(dataset), maxima(dataset)` without doing the computation twice.
"""
function minmaxima(data::AbstractDataset{D, T}) where {D, T<:Real}
    mi = Vector(data[1])
    ma = Vector(data[1])
    for point in data
        for i in 1:D
            if point[i] > ma[i]
                ma[i] = point[i]
            elseif point[i] < mi[i]
                mi[i] = point[i]
            end
        end
    end
    return SVector{D, T}(mi), SVector{D, T}(ma)
end

#####################################################################################
#                                     SVD                                           #
#####################################################################################
using LinearAlgebra
# SVD of Base seems to be much faster when the "long" dimension of the matrix
# is the first one, probably due to Julia's column major structure.
# This does not depend on using `svd` or `svdfact`, both give same timings.
# In fact it is so much faster, that it is *much* more worth it to
# use `Matrix(data)` instead of `reinterpret` in order to preserve the
# long dimension being the first.
"""
    svd(d::AbstractDataset) -> U, S, Vtr
Perform singular value decomposition on the dataset.
"""
function LinearAlgebra.svd(d::AbstractDataset)
    F = svd(Matrix(d))
    return F[:U], F[:S], F[:Vt]
end

#####################################################################################
#                                regularize                                         #
#####################################################################################
using Statistics

"""
    regularize(d::Dataset) → r
Create a regularized version of the input dataset where each timeseries (column)
is transformed to have mean 0 and standard deviation 1.
"""
regularize(d::AbstractDataset) = Dataset(regularized_timeseries(d)[1]...)
function regularized_timeseries(d::AbstractDataset)
    xs = columns(d)
    means = mean.(xs)
    stds = std.(xs)
    for i in 1:length(xs)
        xs[i] .= (xs[i] .- means[i]) ./ stds[i]
    end
    return xs, means, stds
end

"""
    regularize(x::Vector) = (x - mean(x))/std(x)
"""
regularize(x::Vector) = regularize!(copy(x))
regularize!(x::Vector) = (x .= (x .- mean(x))./std(x))
