module KUparser

using InplaceOps
using KUnet

typealias Features Matrix{Int8}
type Sentence form; lemma; cpostag; postag; head; deprel; wvec; Sentence()=new(); end
typealias Corpus AbstractVector{Sentence}
wdim(s)=size(s.wvec,1)
wcnt(s)=size(s.wvec,2)

isdefined(:UInt16) || typealias UInt16 Uint16
isdefined(:UInt8) || typealias UInt16 Uint8

include("archybrid.jl")
include("features.jl")
include("gparser.jl")

end
