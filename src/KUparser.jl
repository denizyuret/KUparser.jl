module KUparser

using InplaceOps
using KUnet
# ?? @everywhere using KUnet

typealias Fmat Matrix{Int8}
type Sentence form; lemma; cpostag; postag; head; deprel; wvec; Sentence()=new(); end
typealias Corpus Vector{Sentence}
wdim(s)=size(s.wvec,1)
wcnt(s)=size(s.wvec,2)

include("archybrid.jl")
include("features.jl")
include("gparser.jl")

# Testing:
include("d1.jl")

end
