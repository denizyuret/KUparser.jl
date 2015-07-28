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

function bparse{T<:Parser}(pt::Type{T}, corpus::Corpus, ndeps::Integer,
                           feats::Fvec, net::Net, nbeam::Integer, 
                           nbatch::Integer=1; xy::Bool=false)
    xy && error("xy not implemented yet.")
    (nbatch < 1 || nbatch > length(corpus)) && (nbatch = length(corpus))
    parses = pt[]
    for s1=1:nbatch:length(corpus)
        s2=min(length(corpus), s1+nbatch-1)                                     
        (batch, fmatrix, score, cost) = initbatch(pt, corpus, s1:s2, ndeps, feats, nbeam)
        nf = 0
        while true
            ### Compute features
            nf1 = nf + 1
            for b in batch
                xy && (b.mincost != 0) && continue
                anyvalidmoves(b.beam[1].parser) || continue
                for bs in b.beam
                    f = features(bs.parser, b.sent, feats, fmatrix, (nf+=1)) # 8258
                    bs.fidx = nf
                end
            end
            nf2 = nf
            nf2 < nf1 && break

            ### Compute scores
            predict(net, sub(fmatrix,:,nf1:nf2), sub(score,:,nf1:nf2); batch=0) # 12783

            ### Update beam
            for b in batch
                xy && (b.mincost != 0) && continue
                anyvalidmoves(b.beam[1].parser) || continue

                ### Compute candidates
                resize!(b.cand,0)
                for bs in b.beam
                    movecosts(bs.parser, b.sent.head, b.sent.deprel, cost) # 670
                    for j=1:length(cost)
                        cost[j] == typemax(Cost) && continue
                        push!(b.cand, BeamState(bs, j, bs.score + score[j,bs.fidx], bs.cost + cost[j])) # 1496
                        # parser and fidx are uninitialized in the candidate
                    end
                end
                @assert (length(b.cand) > 0) "No candidates found"

                ### Sort candidates
                sort!(b.cand; rev=true) # 338

                ### Top candidates to the beam, update mincost
                resize!(b.beam,0)
                b.mincost = typemax(Cost)
                for bc in b.cand
                    length(b.beam) == nbeam && break
                    bc.cost < b.mincost && (b.mincost = bc.cost)
                    bc.parser = copy(bc.parent.parser)     # 2201
                    move!(bc.parser, bc.move)              # 265
                    push!(b.beam, bc)                      # 10
                end
            end # for b in batch
        end # while true
        for b in batch
            push!(parses, b.beam[1].parser)
            # TODO: go back on beams and push x/y using fmatrix and BeamState.move,score
        end
    end # for s1=1:nbatch:length(corpus)
    return parses
end # function bparse

function initbatch{T<:Parser}(pt::Type{T}, corpus::Corpus, r::UnitRange{Int}, ndeps::Integer, feats::DFvec, nbeam::Integer)
    batch = [Beam(pt, corpus[i], ndeps) for i in r]
    s1 = batch[1].sent
    p1 = batch[1].beam[1].parser
    nmove = p1.nmove
    ftype = wtype(s1)
    frows = flen(p1, s1, feats)
    nword = 0; for i in r; nword += wcnt(corpus[i]); end
    fcols = 2 * nword * nbeam
    fmatrix=Array(ftype, frows, fcols)
    score=Array(ftype, nmove, fcols)
    cost=Array(Cost, nmove)
    return (batch, fmatrix, score, cost)
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
