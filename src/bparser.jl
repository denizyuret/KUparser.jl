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
    stop::Bool
    function Beam{T<:Parser}(pt::Type{T}, s::Sentence, ndeps::Integer)
        p = pt(wcnt(s),ndeps)
        b = BeamState(p)
        new(BeamState[b], BeamState[], s, !anyvalidmoves(p))
    end
end

typealias Batch Vector{Beam}

# Life cycle of a Beam:
# 1. bp_resize!: beam[] (one element, no fidx), cand[] empty
# 2. bp_features: beam[i] gets fidx
# 3. predict: score matrix filled
# 4. bp_update_cand: cand (no parser/fidx) filled with children of beam, sorted
# 5. bp_earlystop: cand[1:nbeam] does not have a 0-cost element stop here
# 6. bp_update_beam: beam filled with top nbeam cand, parsers added
# 7. anyvalidmoves: if beam[1] has no valid moves stop here
# 8. goto step 2

function bparse_pmap{T<:Parser}(pt::Type{T}, corpus::Corpus, ndeps::Int,
                                feats::DFvec, net::Net, nbeam::Int;
                                nbatch::Int=1, xy::Bool=false)
    d = distribute(corpus)
    net = testnet(net)
    @date "pmap Starting"
    p = pmap(procs(d)) do x
        bparse(pt, localpart(d), ndeps, feats, gpucopy(net), nbeam; nbatch=nbatch, xy=xy)
    end
    @date "pmap Finished"
    return p
end

function bparse{T<:Parser}(pt::Type{T}, corpus::Corpus, ndeps::Int,
                           feats::DFvec, net::Net, nbeam::Int;
                           nbatch::Int=1, xy::Bool=false)
    @date "bparse Starting"
    (nbatch < 1 || nbatch > length(corpus)) && (nbatch = length(corpus))
    (parses, batch, fmatrix, score) = bp_init(pt, corpus, ndeps, feats, xy)
    for s1=1:nbatch:length(corpus)
        s2=min(length(corpus), s1+nbatch-1)                                     
        bp_resize!(pt, corpus, s1:s2, ndeps, nbeam, batch, fmatrix, score) # :86
        nf = 0
        while true
            nf1 = nf + 1
            nf2 = nf = bp_features(batch, feats, fmatrix, nf) # :35152
            nf2 < nf1 && break
            predict(net, sub(fmatrix,:,nf1:nf2), sub(score,:,nf1:nf2); batch=0) # :26933
            bp_update(batch, score, nbeam, xy) # :10750
        end
        # @show nf
        bp_result(parses, batch, fmatrix, xy) # :1
    end # for s1=1:nbatch:length(corpus)
    @date "bparse Finished"
    return (xy ? (parses[1], copy(parses[2].arr), copy(parses[3].arr)) : parses)
end # function bparse

function bp_init{T<:Parser}(pt::Type{T}, corpus::Corpus, ndeps::Int, feats::DFvec, xy::Bool)
    batch = Array(Beam, 0)
    s0 = corpus[1]
    p0 = pt(wcnt(s0),ndeps)
    fmatrix = KUdense(Array, wtype(corpus), (flen(p0,s0,feats), 0))
    score = KUdense(Array, wtype(corpus), (p0.nmove, 0))
    parses = (xy ? (pt[], copy(fmatrix), copy(score)) : pt[])
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
    # @show fcols
    resize!(fmatrix, (size(fmatrix,1), fcols))
    resize!(score, (size(score,1), fcols))
end

function bp_features(batch::Batch, feats::DFvec, fmatrix::KUdense, nf::Int)
    for s in batch
        s.stop && continue
        for bs in s.beam
            size(fmatrix,2) > nf || error("Out of bounds")
            f = features(bs.parser, s.sent, feats, fmatrix.arr, (nf+=1)) # :35029
            bs.fidx = nf
        end
    end
    return nf
end

function bp_update(batch::Batch, score::KUdense, nbeam::Int, xy::Bool)
    for s in batch
        s.stop && continue
        bp_update_cand(s, score) # :5128
        xy && (s.stop = bp_earlystop(s,nbeam)) && continue
        bp_update_beam(s, nbeam) # :5604
        (s.stop = !anyvalidmoves(s.beam[1].parser)) && continue
    end
end

function bp_update_cand(s::Beam, score::KUdense)
    resize!(s.cand,0)
    cost = Array(Cost, size(score,1))
    for bs in s.beam
        movecosts(bs.parser, s.sent.head, s.sent.deprel, cost) # 162
        for j=1:length(cost)
            cost[j] == typemax(Cost) && continue
            push!(s.cand, BeamState(bs, j, bs.score + score.arr[j,bs.fidx], bs.cost + cost[j])) # :4119
        end
    end
    (length(s.cand) > 0) || error("No candidates found")
    sort!(s.cand; rev=true) # 339
end

function bp_earlystop(s::Beam, nbeam::Int)
    for i=1:min(nbeam,length(s.cand))
        s.cand[i].cost == 0 && (return false)
    end
    return true
end

function bp_update_beam(s::Beam, nbeam::Int)
    resize!(s.beam,0)
    for i=1:min(nbeam, length(s.cand))
        bc = s.cand[i]
        bc.parser = copy(bc.parent.parser)     # :5480
        move!(bc.parser, bc.move)              # 20
        push!(s.beam, bc)                      # 3
    end
end

function bp_result(parses, batch::Batch) # no xy version
    for s in batch 
        push!(parses, s.beam[1].parser)
    end
end

function bp_result(parses, batch::Batch, fmatrix::KUdense, xy::Bool) # xy version
    xy || (return bp_result(parses,batch))
    (p, x, y) = parses
    (mx,nx) = size(x)
    nf = size(fmatrix,2)
    resize!(x, (size(x,1), nx+nf))
    resize!(y, (size(y,1), nx+nf))
    f2x = zeros(Int, nf) # f2x[fidx] => xidx
    grad = zeros(Float64,0)
    for s in batch
        push!(p, s.beam[1].parser)
        bp_softmax(s, grad)
        foundgold = false       # in case there is more than one, only treat the highest scoring zero cost answer as gold
        for i=1:length(s.cand)
            bs = s.cand[i]
            bs.cost == 0 && !foundgold && (foundgold=true; grad[i] -= 1)
            while bs.parent != NullBeamState
                fidx = bs.parent.fidx
                xidx = f2x[fidx]
                if xidx == 0
                    f2x[fidx] = xidx = (nx += 1)
                    nx <= size(x,2) || error("Out of bounds")
                    copy!(x, (xidx-1)*mx+1, fmatrix, (fidx-1)*mx+1, mx)
                    y[:,xidx] = zero(eltype(y))
                end
                y[bs.move,xidx] += grad[i]
                bs = bs.parent
            end
        end
        # s.sent.form[1]=="Influential" && print_beam(s, y.arr, f2x)
    end
    # @show nx
    resize!(x, (size(x,1), nx))
    resize!(y, (size(y,1), nx))
end

function bp_softmax(s::Beam, prob::Vector{Float64})
    smax = typemin(Float64)
    z = zero(Float64)
    nc = length(s.cand)
    resize!(prob, nc)
    for bc in s.cand; bc.score > smax && (smax = bc.score); end
    for i=1:nc; prob[i] = exp(s.cand[i].score - smax); z += prob[i]; end
    for i=1:nc; prob[i] /= z; end
end

function print_beam(s::Beam, y::Matrix, f2x::Vector{Int})
    for i = 1:length(s.beam)
        bs = s.cand[i]
        ba = Array(BeamState,0)
        while bs.parent != NullBeamState
            unshift!(ba, bs)
            bs = bs.parent
        end
        for bs in ba
            parent_id = isdefined(bs.parent,:parser) ? hash(bs.parent.parser) & 0xff : 0
            parser_id = isdefined(bs,:parser) ? hash(bs.parser) & 0xff : 0
            grad = y[bs.move, f2x[bs.parent.fidx]]
            @printf "%02x/%d/%.2f/%+.2f/%02x " parent_id bs.cost bs.score grad parser_id 
            # print("$(bs.score)/$(bs.cost)/$(bs.grad) ")
            # @printf "%.2f/%d/%d/%d/%+f " bs.score bs.cost bs.fidx bs.parent.fidx y[bs.move, f2x[bs.parent.fidx]]
            # print("$(bs.fidx) ")
            # print("$(pointer(bs.parser.head)) ")
            # bs = bs.parent
        end
        println()
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

# function bp_pathsum(s::Beam, fcols::Vector{Int8})
#     ncols = 0
#     for bs in s.beam
#         bs.cost == 0 && (bs.grad -= 1) # gradient is p for bad paths, p-1 for good paths
#         ps = bs.parent
#         while ps != NullBeamState
#             ps.grad += bs.grad
#             ps = ps.parent
#         end
#     end
#     return ncols
# end

        # first resize x/y and init f2x, zero y, copy to x, then do grad sum directly on y
        # dont count, use final shrink, just make sure we have enough space to fit all fmatrix columns

        # ncols = bp_pathsum(s, fcols)
        # print_beam(s); println()
        # we need to put bs.parent.fidx into x, y[bs.move]+=bs.grad, multiple children can update same y

    # if xy
    #     @show map(typeof, parses)
    #     @show map(size, parses)
    #     parses = (parses[1], parses[2].arr, parses[3].arr)
    # end
