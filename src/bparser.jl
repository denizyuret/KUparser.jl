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

type Beam beam; beam2; cand; sent; nbeam
    function Beam(p::Parser, s::Sentence, ndeps::Integer, feats::Fvec, net::Net, nbeam::Integer)
        b = new()
        nword = wcnt(s)
        ptype = typeof(p)
        nmove = p.nmove
        ncand = nbeam * p.nmove

        b.nbeam = 1                                     # number of parsers: b.nbeam is actual, nbeam is max
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
    BeamState(ptype, nword, ndeps, nmove)=new(ptype(nword,ndeps), zero(Float64), zero(Cost), Array(Cost, nmove))
end

# Each BeamCandidate represents a potential move from a parent state and its cumulative score

type BeamCandidate 
    score::Float64      # cumulative score if we execute move in parent
    parent::BeamState   # parent state
    move::Move          # move to be executed
    BeamCandidate()=new()
end


# Here is the workhorse:
function bparse{T<:Parser}(p::Vector{T}, corpus::Corpus, ndeps::Integer, feats::Fvec, net::Net, nbeam::Integer, nbatch::Integer,
                           x::AbstractArray=[], y::AbstractArray=[], nx::Integer=0)
    @assert isa(net[end], LogpLoss) "Need LogpLoss final layer for beam parser"
    (nbatch == 0 || nbatch > length(corpus)) && (nbatch = length(corpus))
    ftype = wtype(corpus[1])
    frows = flen(p[1], corpus[1], feats)
    fcols = nbeam * nbatch
    f = Array(ftype, frows, fcols)
    nmove = p[1].nmove
    score = Array(ftype, nmove, fcols)
    dbg = Dict();

    for s1=1:nbatch:length(corpus)                                              # processing corpus[s1:s2]
        s2=min(length(corpus), s1+nbatch-1)                                     
        batch = [Beam(p[i], corpus[i], ndeps, feats, net, nbeam) for i=s1:s2]   # initialize beam for each sentence
        while true                                              
            nf = 0                                                              # f[1:nf]: feature vectors for the whole batch s1:s2
            for b in batch                                                      # b is the beam (multiple parser states) for one sentence in s1:s2
                anyvalidmoves(b.beam[1].parser) || continue                     # assuming all parsers for a sentence finish at the same time
                (cmin,jmin,fmin) = (Pinf,0,0)                                   # mincost, its move, its f index
                st = b.sent                                                     # sentence for this beam
                for i=1:b.nbeam
                    bs = b.beam[i]                                              # bs is a parser state on b
                    features(bs.parser, st, feats, f, (nf+=1))                  # f[:,nf] is the feature vector for bs.parser
                    movecosts(bs.parser, st.head, st.deprel, bs.mcost)          # bs.mcost[j] is the cost of move j
                    for j=1:nmove
                        ccost = bs.mcost[j] + bs.cost                           # cumulative cost of move j
                        if ccost < cmin
                            dbg[:yp] = i
                            dbg[:ym] = j
                            dbg[:yc] = bs.mcost[j]
                            dbg[:ycc] = ccost
                        end
                        ccost < cmin && ((cmin,jmin,fmin)=(ccost,j,nf))         # record mincost state and move
                    end
                end # for i=1:b.nbeam

                @assert (cmin < Pinf)
                if !isempty(x)                                                  # if asked for x, copy mincost column from f
                    nx += 1                                                     # implement early stop here?
                    copy!(x, (nx-1)*frows+1, f, (fmin-1)*frows+1, frows)        
                    y[:,nx]=zero(eltype(y))
                    y[jmin,nx]=one(eltype(y))                                   # and set mincost y move to 1
                end
            end # for b in batch (1)
            nf == 0 && break                                                    # no more valid moves for any sentence
            predict(net, sub(f,:,1:nf), sub(score,:,1:nf); batch=0)             # scores in score[1:nf]
            nf1 = nf; nf = 0                                                    # nf will count 1..nf1 during second pass
            for b in batch                                                      # collect candidates in second pass
                anyvalidmoves(b.beam[1].parser) || continue
                nc = 0                                                          # nc is number of candidates for sentence beam b
                for i=1:b.nbeam
                    bs = b.beam[i]
                    @assert any(bs.mcost .< Pinf)
                    nf += 1                                                     # score[j,nf] should be the score for sentence b, parser i, move j
                    for j=1:nmove                                               
                        bs.mcost[j] == Pinf && continue                         # bs.mcost[j] is the cost of move j
                        nc += 1                                                 # add new candidate
                        b.cand[nc].parent = bs                                  # parent state
                        b.cand[nc].move = j                                     # candidate move
                        b.cand[nc].score = bs.score + score[j,nf]               # cumulative score for state + move

                        if i==dbg[:yp] && j==dbg[:ym]
                            dbg[:ys] = b.cand[nc].score - bs.score
                            dbg[:yss] = b.cand[nc].score
                        end
                    end
                end # for i=1:b.nbeam
                @assert (nc > 0) "No candidates found"
                sort!(sub(b.cand,1:nc); rev=true)
                b.nbeam = min(nc,nbeam)                                         # b.nbeam is now the new beam size
                for i=1:b.nbeam
                    bc = b.cand[i]
                    bs = bc.parent
                    copy!(b.beam2[i].parser, bs.parser)
                    move!(b.beam2[i].parser, bc.move)
                    b.beam2[i].score = bc.score
                    b.beam2[i].cost = bs.cost + bs.mcost[bc.move]

                    if i==1
                        dbg[:zp] = findfirst(b.beam, bs)
                        dbg[:zm] = bc.move
                        dbg[:zc] = bs.mcost[bc.move]
                        dbg[:zcc] = bs.cost + bs.mcost[bc.move]
                        dbg[:zs] = bc.score - bs.score
                        dbg[:zss] = bc.score
                    end

                end # for i=1:b.nbeam
                dbgprint(dbg, b)

                b.beam,b.beam2 = b.beam2,b.beam

            end # for b in batch (2)
            @assert (nf == nf1) "$nf != $nf1"
        end # while true
        for s=s1:s2
            copy!(p[s], batch[s-s1+1].beam[1].parser)                           # copy the best parses to output

            ss = corpus[s]
            for i=1:wcnt(ss); print("$(ss.form[i])($i) "); end; println("")
            println(map(int, ss.head))
            println(map(int, batch[s-s1+1].beam[1].parser.head))
            # b = batch[s-s1+1]
            # for i=1:b.nbeam
            #     bs = b.beam[i]
            #     @assert (truecost(bs.parser, b.sent) == int(bs.cost))
            # end
        end
    end # for s1=1:nbatch:length(corpus)
end # function bparse

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

# Need this so sort works:
Base.isless(a::BeamCandidate,b::BeamCandidate)=(a.score < b.score)


# Single cpu version:
function bparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer, feats::Fvec, net::Net, 
                           nbeam::Integer, nbatch::Integer=1; xy::Bool=false)
    pa = map(s->pt(wcnt(s), ndeps), c)
    if xy
        xtype = wtype(c[1])
        x = Array(xtype, xsize(pa[1], c, feats))
        y = zeros(xtype, ysize(pa[1], c))
        bparse(pa, c, ndeps, feats, net, nbeam, nbatch, x, y)
        return (pa,x,y)
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
