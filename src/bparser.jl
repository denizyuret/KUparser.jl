# The public interface for bparse takes the following arguments:
#
# function bparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer, feats::Fvec, net::Net, 
#                            nbeam::Integer, nbatch::Integer, ncpu::Integer; xy::Bool=false)
#
# pt::ParserType: ArcHybridR1, ArcEagerR1, ArcHybrid13, ArcEager13
# c::Corpus: array of input sentences
# ndeps::Integer: number of dependency types
# feats::Fvec: specification of features
# net::Net: model used for move prediction
# nbeam::Integer: beam size
# nbatch::Integer: (optional) parse sentences in batches for efficiency
# ncpu::Integer: (optional) perform parallel processing
# xy::Bool: (keyword) return (p,x,y) tuple for training, by default only parsers returned.

# Data structures for beam search:

# A batch consists of multiple sentences and their beams.
# A beam consists of an array of BeamState's and an array of BeamCandidate's.
# The BeamStates and the BeamCandidates in a beam are sorted by cumulative score.
# We also track the mcs, index of the mincoststate for training and early stop.

type Beam beam; beam2; cand; sent; nbeam; mcs
    function Beam(p::Parser, s::Sentence, ndeps::Integer, feats::Fvec, net::Net, nbeam::Integer)
        b = new()
        nword = wcnt(s)
        ptype = typeof(p)
        nmove = p.nmove
        ncand = nbeam * p.nmove

        b.nbeam = 1                                     # number of parsers: b.nbeam is actual, nbeam is max
        b.mcs = 1                                       # index of the mincoststate
        b.sent = s                                      # sentence of the beam
        b.beam  = [BeamState(ptype, nword, ndeps, nmove) for i=1:nbeam]
        b.beam2 = [BeamState(ptype, nword, ndeps, nmove) for i=1:nbeam]         # for swap
        b.cand  = [BeamCandidate() for i=1:ncand]
        return b
    end
end

# Each BeamState is a parser state with its cumulative score and cost and current move costs

type BeamState 
    parser::Parser 	# parser state
    score::Float64      # cumulative score (logp)
    cost::Cost          # cumulative cost (gold arcs that have become impossible)
    mcost::Vector{Cost} # move costs from this state
    fidx::Int           # column index in feature matrix
    BeamState(ptype, nword, ndeps, nmove)=new(ptype(nword,ndeps), zero(Float64), zero(Cost), Array(Cost, nmove))
end

# Each BeamCandidate represents a potential move from a parent state and its cumulative score

type BeamCandidate 
    state::BeamState    # parent state
    move::Move          # move to be executed
    score::Float64      # cumulative score if we execute move in state
    BeamCandidate()=new()
end

# BeamDebug = false

# Here is the workhorse:
function bparse{T<:Parser}(p::Vector{T}, corpus::Corpus, ndeps::Integer, feats::Fvec, net::Net, nbeam::Integer, nbatch::Integer,
                           x::AbstractArray=[], y::AbstractArray=[], nx::Integer=0)
    # Do not locally normalize!
    # @assert isa(net[end], LogpLoss) "Need LogpLoss final layer for beam parser"
    (nbatch == 0 || nbatch > length(corpus)) && (nbatch = length(corpus))
    ftype = wtype(corpus[1])
    frows = flen(p[1], corpus[1], feats)
    fcols = nbeam * nbatch
    f = Array(ftype, frows, fcols)
    nmove = p[1].nmove
    score = Array(ftype, nmove, fcols)
    # BeamDebug && (dbg = Dict())

    for s1=1:nbatch:length(corpus)                                              # processing corpus[s1:s2]
        s2=min(length(corpus), s1+nbatch-1)                                     
        batch = [Beam(p[i], corpus[i], ndeps, feats, net, nbeam) for i=s1:s2]   # initialize beam for each sentence
        while true                                              

            # Compute move scores in bulk to take advantage of the GPU
            nf = 0                                                              # f[1:nf]: feature vectors for the whole batch s1:s2
            for b in batch                                                      # b is the beam (multiple parser states) for one sentence in s1:s2
                !isempty(x) && (b.beam[b.mcs].cost != 0) && continue            # early stop if training and zero cost state falls out of beam
                anyvalidmoves(b.beam[1].parser) || continue                     # parsing finished, assuming all parsers for a sentence finish at the same time
                for i=1:b.nbeam                                                 # b.beam[1:b.nbeam] are valid parser states
                    bs = b.beam[i]
                    bs.fidx = (nf+=1)
                    features(bs.parser, b.sent, feats, f, bs.fidx)              # f[:,bs.fidx] is the feature vector for bs.parser
                end
            end
            nf == 0 && break                                                    # no more valid moves for any sentence
            predict(net, sub(f,:,1:nf), sub(score,:,1:nf); batch=0)             # scores in score[1:nf]

            for b in batch                                                      
                !isempty(x) && (b.beam[b.mcs].cost != 0) && continue
                anyvalidmoves(b.beam[1].parser) || continue
                nc = 0                                                          # nc is number of candidates for sentence beam b
                for i=1:b.nbeam                                                 # collect candidates in second pass
                    bs = b.beam[i]
                    movecosts(bs.parser, b.sent.head, b.sent.deprel, bs.mcost)
                    for j=1:nmove                                               
                        bs.mcost[j] == Pinf && continue                         # bs.mcost[j] is the cost of move j
                        bc = b.cand[nc += 1]                                    # add new candidate
                        bc.state = bs                                           # parent state
                        bc.move = j                                             # candidate move
                        bc.score = bs.score + score[j,bs.fidx]                  # cumulative score for state + move
                    end
                end
                @assert (nc > 0) "No candidates found"

                #@show (nc, nbeam, b.nbeam)
                #display(b.beam[1:b.nbeam])

                sort!(sub(b.cand,1:nc); rev=true)                               # sort candidates by score
                b.mcs = 1                                                       # reset mincoststate
                b.nbeam = min(nc,nbeam)                                         # b.nbeam is now the new beam size
                for i=1:b.nbeam                                                 # fill b.beam2[1:b.nbeam] from b.cand[1:b.nbeam]
                    bc = b.cand[i]
                    b1 = bc.state
                    b2 = b.beam2[i]
                    copy!(b2.parser, b1.parser)
                    move!(b2.parser, bc.move)
                    b2.score = bc.score
                    b2.cost = b1.cost + b1.mcost[bc.move]
                    (b2.cost < b.beam2[b.mcs].cost) && (b.mcs = i)
                end

                # c1 = ctuple(b,1)
                # c2 = ctuple(b, findfirst(b.cand,c0))
                # y0 = findfirst(bc->(bc.state.cost + bc.state.mcost[bc.move]==0), b.cand[1:nc])
                # c3 = (y0==0 ? y0 : ctuple(b, y0))
                # c4 = ctuple(b, b.nbeam)
                # println("")
                # @show c1
                # @show c2
                # @show c3
                # @show c4

                if (!isempty(x) &&
                    (b.mcs != 1) &&
                    (b.beam2[b.mcs].cost == 0))
                    c0 = b.cand[b.mcs]
                    c1 = b.cand[1]
                    addxy(x, (nx+=1), f, c0.state.fidx, y, c0.move, -one(eltype(y)))
                    addxy(x, (nx+=1), f, c1.state.fidx, y, c1.move, one(eltype(y)))
                end

                b.beam,b.beam2 = b.beam2,b.beam

            end # for b in batch (2)
        end # while true
        for s=s1:s2
            copy!(p[s], batch[s-s1+1].beam[1].parser)                           # copy the best parses to output

            # if BeamDebug
            #     ss = corpus[s]
            #     for i=1:wcnt(ss); print("$(ss.form[i])($i) "); end; println("")
            #     println(map(int, ss.head))
            #     println(map(int, batch[s-s1+1].beam[1].parser.head))
            #     b = batch[s-s1+1]
            #     for i=1:b.nbeam
            #         bs = b.beam[i]
            #         @assert (truecost(bs.parser, b.sent) == int(bs.cost))
            #     end
            # end
        end
    end # for s1=1:nbatch:length(corpus)
    return nx
end # function bparse

function ctuple(b, i)
    bc = b.cand[i]
    bs = bc.state
    bi = findfirst(b.beam, bs)
    move = int(bc.move)
    cost = int(bs.cost + bs.mcost[bc.move])
    score = int(1000 * bc.score)
    (i, bi, move, cost, score, bs.parser)
end

function addxy(x, nx, f, nf, y, yidx, yval)
    frows = size(f,1)
    copy!(x, (nx-1)*frows+1, f, (nf-1)*frows+1, frows)
    y[:,nx]=zero(eltype(y))
    y[yidx,nx]=yval
end

# Need this so sort works:
Base.isless(a::BeamCandidate,b::BeamCandidate)=(a.score < b.score)


# Single cpu version:
function bparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer, feats::Fvec, net::Net, 
                           nbeam::Integer, nbatch::Integer=1; xy::Bool=false)
    pa = map(s->pt(wcnt(s), ndeps), c)
    if xy
        xtype = wtype(c[1])
        xdims = xsize(pa[1], c, feats)
        ydims = ysize(pa[1], c)
        x = Array(xtype, xdims[1], 2*xdims[2]) # we can have two training instances for each move
        y = zeros(xtype, ydims[1], 2*ydims[2])
        nx = bparse(pa, c, ndeps, feats, net, nbeam, nbatch, x, y)
        return (pa, sub(x,:,1:nx), sub(y,:,1:nx))
    else
        bparse(pa, c, ndeps, feats, net, nbeam, nbatch)
        return pa
    end
end

# Multi cpu version:
function bparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer, feats::Fvec, net::Net, 
                           nbeam::Integer, nbatch::Integer, ncpu::Integer; xy::Bool=false)
    d = distribute(c)
    net = testnet(net)
    pmap(procs(d)) do x
        bparse(pt, localpart(d), ndeps, feats, gpucopy(net), nbeam, nbatch; xy=xy)
    end
end




function dbgprint(dbg, b)
    @printf("yp=%02d ", dbg[:yp])
    @printf("ym=%02d ", dbg[:ym])
    @printf("yc=%d ", dbg[:yc])
    @printf("ycc=%d ", dbg[:ycc])
    @printf("ys=%.2e ", dbg[:ys])
    @printf("yss=%.2e ", dbg[:yss])
    s = b.sent
    p = b.beam[dbg[:yp]].parser
    w0 = (p.wptr <= wcnt(s) ? s.form[p.wptr] : :none)
    s0 = (p.sptr >= 1 ? s.form[p.stack[p.sptr]] : :none)
    mv = (dbg[:ym]==1 ? "X" :
          dbg[:ym]==p.nmove ? "S" :
          dbg[:ym]%2==0 ? "L" : "R")
    println("[$s0|$w0]:$mv")
    @printf("zp=%02d ", dbg[:zp])
    @printf("zm=%02d ", dbg[:zm])
    @printf("zc=%d ", dbg[:zc])
    @printf("zcc=%d ", dbg[:zcc])
    @printf("zs=%.2e ", dbg[:zs])
    @printf("zss=%.2e ", dbg[:zss])
    p = b.beam[dbg[:zp]].parser
    w0 = (p.wptr <= wcnt(s) ? s.form[p.wptr] : :none)
    s0 = (p.sptr >= 1 ? s.form[p.stack[p.sptr]] : :none)
    mv = (dbg[:zm]==1 ? "X" :
          dbg[:zm]==p.nmove ? "S" :
          dbg[:zm]%2==0 ? "L" : "R")
    println("[$s0|$w0]:$mv\n")
end

