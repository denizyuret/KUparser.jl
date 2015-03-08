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

include("archybrid.jl")
include("features.jl")
include("gparser.jl")

end
