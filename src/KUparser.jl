module KUparser

using InplaceOps
using KUnet

typealias Fmat Matrix{Int8}
type Sentence form; lemma; cpostag; postag; head; deprel; wvec; Sentence()=new(); end
typealias Corpus Vector{Sentence}

include("archybrid.jl")
include("features.jl")
include("gparser.jl")

# Testing:
include("d1.jl")

end
