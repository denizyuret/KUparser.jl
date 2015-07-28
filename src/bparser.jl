type BeamState 
    parent::BeamState   # previous state
    move::Move          # move executed to get to this state
    score::Float64      # cumulative score (logp)
    cost::Cost          # cumulative cost (gold arcs that have become impossible)
    fidx::Int           # column in feature matrix representing current parser state
    parser::Parser 	# current parser state
    BeamState()=new()
    BeamState(p::Parser)=new(NullBeamState,0,0,0,0,p)
    BeamState(b::BeamState,m::Integer,s::Number,c::Integer)=
        new(b,convert(Move,m),convert(Float64,s),convert(Cost,c))
end

const NullBeamState = BeamState()
Base.isless(a::BeamState,b::BeamState)=(a.score < b.score)

type Beam 
    beam::Vector{BeamState}
    cand::Vector{BeamState}
    sent::Sentence
    mincost::Cost
    function Beam{T<:Parser}(pt::Type{T}, s::Sentence, ndeps::Integer)
        p = pt(wcnt(s),ndeps)
        b = BeamState(p)
        new(BeamState[b], BeamState[], s, 0)
    end
end

typealias Batch Vector{Beam}



function bparse{T<:Parser}(pt::Type{T}, corpus::Corpus, ndeps::Int,
                           feats::DFvec, net::Net, nbeam::Int, 
                           nbatch::Int=1; xy::Bool=false)
    xy && error("xy not implemented yet.")
    (nbatch < 1 || nbatch > length(corpus)) && (nbatch = length(corpus))
    (parses, batch, fmatrix, score) = bp_init(pt, corpus, ndeps, feats)
    for s1=1:nbatch:length(corpus)
        s2=min(length(corpus), s1+nbatch-1)                                     
        bp_resize!(pt, corpus, s1:s2, ndeps, nbeam, batch, fmatrix, score) # 34
        nf = 0
        while true
            nf1 = nf + 1
            nf2 = bp_features(batch, feats, fmatrix.arr, nf, xy) # 6431
            nf2 < nf1 && break
            predict(net, sub(fmatrix,:,nf1:nf2), sub(score,:,nf1:nf2); batch=0) # 12785
            for b in batch
                xy && (b.mincost != 0) && continue
                anyvalidmoves(b.beam[1].parser) || continue
                bp_update_cand(b, score.arr) # 568
                sort!(b.cand; rev=true) # 339
                bp_update_beam(b, nbeam) # 369
            end
        end
        for b in batch
            push!(parses, b.beam[1].parser)
            # TODO: go back on beams and push x/y using fmatrix and BeamState.move,score
        end
    end # for s1=1:nbatch:length(corpus)
    return parses
end # function bparse

function bp_init{T<:Parser}(pt::Type{T}, corpus::Corpus, ndeps::Int, feats::DFvec)
    parses = pt[]
    batch = Array(Beam, 0)
    s0 = corpus[1]
    p0 = pt(wcnt(s0),ndeps)
    fmatrix = KUdense(Array, wtype(corpus), (flen(p0,s0,feats), 0))
    score = KUdense(Array, wtype(corpus), (p0.nmove, 0))
    return (parses, batch, fmatrix, score)
end

function bp_resize!{T<:Parser}(pt::Type{T}, corpus::Corpus, r::UnitRange{Int}, ndeps::Int, nbeam::Int, 
                               batch::Batch, fmatrix::KUdense, score::KUdense)
    batch = resize!(batch, length(r))
    nword = 0
    j = 0
    for i in r
        batch[j+=1] = Beam(pt, corpus[i], ndeps)
        nword += wcnt(corpus[i])
    end
    fcols = 2*nword*nbeam
    resize!(fmatrix, (size(fmatrix,1), fcols))
    resize!(score, (size(score,1), fcols))
end

function bp_features(batch::Batch, feats::DFvec, fmatrix::Matrix, nf::Int, xy::Bool)
    for b in batch
        xy && (b.mincost != 0) && continue
        anyvalidmoves(b.beam[1].parser) || continue
        for bs in b.beam
            f = features(bs.parser, b.sent, feats, fmatrix, (nf+=1)) # 6589
            bs.fidx = nf
        end
    end
    return nf
end

function bp_update_cand(b::Beam, score::Matrix)
    resize!(b.cand,0)
    cost = Array(Cost, size(score,1))
    for bs in b.beam
        movecosts(bs.parser, b.sent.head, b.sent.deprel, cost) # 162
        for j=1:length(cost)
            cost[j] == typemax(Cost) && continue
            push!(b.cand, BeamState(bs, j, bs.score + score[j,bs.fidx], bs.cost + cost[j])) # 1242
        end
    end
    (length(b.cand) > 0) || error("No candidates found")
end

function bp_update_beam(b::Beam, nbeam::Int)
    resize!(b.beam,0)
    b.mincost = typemax(Cost)
    for bc in b.cand
        length(b.beam) == nbeam && break
        bc.cost < b.mincost && (b.mincost = bc.cost)
        bc.parser = copy(bc.parent.parser)     # 378
        move!(bc.parser, bc.move)              # 20
        push!(b.beam, bc)                      # 3
    end
end

# Data structures for beam search:

# A batch consists of multiple sentences and their beams.
# A Beam consists of a sentence, a beam array and a cand array.
# Both arrays consist of BeamStates sorted by cumulative score.
# We also track the mincost for training and early stop.

# Say we have a n sentence corpus.
# On return we give back n parsers.
# If training we give back x and y.  The size of x and y are uncertain.
# During processing we need a beam for each sentence of size nbeam.
# The beam stores BeamState's.  We allocate BeamState's dynamically.
# Internally we process sentences in batches of size nbatch.
# We need to keep whole of f, no reusing, and copy part of it to return x.
# We could use KUdense or KUsparse for f and use resize... that supports sparse as well.
# that would mean modifying features.jl and sfeatures.jl to use that too.
# calling convention for sparse and dense features.jl are not the same now.
# we could have both append a column at the end of a given array?
# or just have them return the feats vector and we do the appending here using ccat.

# beam: nbeam, beam, sent, mcs, cand, beam2; do we need?
# prealloc parsers but no prealloc of beamstates? parsers dynamic too.


### DEAD CODE


#     batch = [Beam(pt, corpus[i], ndeps) for i in r]
#     s1 = batch[1].sent
#     p1 = batch[1].beam[1].parser
#     nmove = p1.nmove
#     ftype = wtype(s1)
#     frows = flen(p1, s1, feats)
#     nword = 0; for i in r; nword += wcnt(corpus[i]); end
#     fcols = 2 * nword * nbeam
#     fmatrix=Array(ftype, frows, fcols)
#     score=Array(ftype, nmove, fcols)
#     cost=Array(Cost, nmove)
#     return (batch, fmatrix, score, cost)
# end

