typealias WordId Int32          # [1:nvocab]
typealias DepRel UInt8          # [1:ndeps]
typealias PosTag UInt8          # [1:npostag]
typealias Position Int32        # [1:nword]
typealias Cost Int32            # [0:nword]
typealias Move Int              # [1:nmove]
typealias SFtype Int32          # representing sparse feature in x
typealias WVtype Float32        # element type for word vectors
typealias Pvec Vector{Position} # used for stack, head
typealias Dvec Vector{DepRel}   # used for deprel
Pzeros(n::Integer...)=zeros(Position, n...)
Droots(n::Integer...)=ones(DepRel, n...) # ROOT=1

type Vocab
    words::Vector{String}
    postags::Vector{String}
    deprels::Vector{String}
    wvecs::Matrix{WVtype}
    fdict::Dict{Any,SFtype}
    Vocab()=new(String[],String[],String[],Array(WVtype,0,0),Dict{Any,SFtype}())
end

type Sentence
    word::Vector{WordId}
    postag::Vector{PosTag}
    head::Vector{Position}
    deprel::Vector{DepRel}
    vocab::Vocab
    # wvec::Array{Float32}
    # TODO: consider using lc(word), lemma, xpostag, feats, deps, misc, lang-specific deprels ext
    # TODO: add word vectors, vocab, how do we add context vectors?
    Sentence()=new(WordId[],PosTag[],Position[],DepRel[])
end

typealias Corpus AbstractVector{Sentence}

wdim(s::Sentence)=size(s.vocab.wvecs,1)
wcnt(s::Sentence)=length(s.word)
wtype(s::Sentence)=eltype(s.vocab.wvecs)
wdim(c::Corpus)=wdim(c[1])
wcnt(c::Corpus)=(n=0;for s in c; n+=wcnt(s); end; n)
wtype(c::Corpus)=wtype(c[1])

function Base.show(io::IO, s::Sentence)
    print(io, "[")
    d = s.vocab.words
    print(io, (wcnt(s) <= 6 ?
               join(d[s.word], " ") :
               join([d[s.word[1:3]]..., "…", d[s.word[end-2:end]]...], " ")))
    print(io, "]")
    print(io, map(Int, s.head))
end

