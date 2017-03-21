type Sentence
    form::Vector{String}
    postag::Vector{PosTag}
    head::Vector{Position}
    deprel::Vector{DepRel}
    wvec::Array{Float32}
    # TODO: consider using lc(form), lemma, xpostag, feats, deps, misc, lang-specific deprels ext
    Sentence()=new(String[],PosTag[],Position[],DepRel[],Float32[]) # TODO: should we hardcode Float32?
end

# Deprecated:
typealias Corpus AbstractVector{Sentence}

# This should eventually replace the Corpus type
type Corpus2
    postags::Dict{String,PosTag}
    deprels::Dict{String,DepRel}
    # TODO: add word vectors, vocab, how do we add context vectors?
    sentences::Vector{Sentence}
    Corpus2()=new()
end

wdim(s::Sentence)=size(s.wvec,1)
wcnt(s::Sentence)=size(s.wvec,2)
wtype(s::Sentence)=eltype(s.wvec)
wdim(c::Corpus)=size(c[1].wvec,1)
wcnt(c::Corpus)=(n=0;for s in c; n+=wcnt(s); end; n)
wtype(c::Corpus)=eltype(c[1].wvec)

function Base.show(io::IO, s::Sentence)
    print(io, "[")
    print(io, (wcnt(s) <= 6 ?
               join(s.form, " ") :
               join([s.form[1:3], "…", s.form[end-2:end]], " ")))
    print(io, "]")
    print(io, map(Int, s.head))
end

