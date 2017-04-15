typealias Word String           # represent words with strings instead of finite vocab, we will process chars
typealias PosTag UInt8          # 17 universal part-of-speech tags
typealias DepRel UInt8          # 37 universal dependency relations
typealias Position Int16        # sentence position
typealias Cost Position         # [0:nword]
typealias Move Int              # [1:nmove]
typealias WordId Int32          # [1:nvocab]
typealias Pvec Vector{Position} # used for stack, head
typealias Dvec Vector{DepRel}   # used for deprel

immutable Vocab
    cdict::Dict{Char,Int}       # character dictionary (input)
    idict::Dict{String,Int}     # word dictionary (input, filled in by maptoint)
    odict::Dict{String,Int}     # word dictionary (output)
    sosword::String             # start-of-sentence word (input)
    eosword::String             # end-of-sentence word (input)
    unkword::String             # unknown word (output, input does not have unk)
    sowchar::Char               # start-of-word char
    eowchar::Char               # end-of-word char
    unkchar::Char               # unknown char
    postags::Dict{String,PosTag}
    deprels::Dict{String,DepRel}
end

;                               # CONLLU FORMAT
immutable Sentence              # 1. ID: Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; may be a decimal number for empty nodes.
    word::Vector{Word}          # 2. FORM: Word form or punctuation symbol.
    #stem::Vector{Stem}         # 3. LEMMA: Lemma or stem of word form.
    postag::Vector{PosTag}      # 4. UPOSTAG: Universal part-of-speech tag.
    #xpostag::Vector{Xpostag}   # 5. XPOSTAG: Language-specific part-of-speech tag; underscore if not available.
    #feats::Vector{Feats}       # 6. FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
    head::Vector{Position}      # 7. HEAD: Head of the current word, which is either a value of ID or zero (0).
    deprel::Vector{DepRel}      # 8. DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
    #deps::Vector{Deps}         # 9. DEPS: Enhanced dependency graph in the form of a list of head-deprel pairs.
    #misc::Vector{Misc}         # 10. MISC: Any other annotation.

    wvec::Vector                # word vectors
    fvec::Vector                # forw context vectors
    bvec::Vector                # back context vectors
    vocab::Vocab                # Common repository of symbols for upostag, deprel etc.

    Sentence(v::Vocab)=new([],[],[],[],[],[],[],v)
end
Base.length(s::Sentence)=length(s.word)
typealias Corpus AbstractVector{Sentence}

# Deprecated?
wdim(s::Sentence)=size(s.vocab.wvecs,1)
wcnt(s::Sentence)=length(s.word)
wtype(s::Sentence)=eltype(s.wvecs)
wdim(c::Corpus)=wdim(c[1])
wcnt(c::Corpus)=(n=0;for s in c; n+=wcnt(s); end; n)
wtype(c::Corpus)=wtype(c[1])

function Base.show(io::IO, s::Sentence)
    print(io, "[")
    print(io, (wcnt(s) <= 6 ?
               join(s.word, " ") :
               join([s.word[1:3]..., "…", s.word[end-2:end]...], " ")))
    print(io, "]")
    print(io, map(Int, s.head))
end
