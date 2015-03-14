module KUparser

using InplaceOps
using KUnet
using Compat
if VERSION >= v"0.4.0-dev+3184"
    using DistributedArrays
end

typealias Features Matrix{Int8}
type Sentence form; lemma; cpostag; postag; head; deprel; wvec; Sentence()=new(); end
typealias Corpus AbstractVector{Sentence}
wdim(s)=size(s.wvec,1)
wcnt(s)=size(s.wvec,2)

# Type representing feature values
typealias Fval Float32
typealias Fvec Vector{Fval}
typealias Fmat Matrix{Fval}

include("archybrid.jl")
include("features.jl")
include("gparser.jl")
include("bparser.jl")

# this has been fixed in Julia 0.4:
import Base: sortperm!, Algorithm, Ordering, DEFAULT_UNSTABLE, Forward, Perm, ord
function sortperm!{I<:Integer}(x::AbstractVector{I}, v::AbstractVector; alg::Algorithm=DEFAULT_UNSTABLE,
                               lt::Function=isless, by::Function=identity, rev::Bool=false, order::Ordering=Forward,
                               initialized::Bool=false)
    length(x) != length(v) && throw(ArgumentError("Index vector must be the same length as the source vector."))
    !initialized && @inbounds for i = 1:length(v); x[i] = i; end
    sort!(x, alg, Perm(ord(lt,by,rev,order),v))
end

end
