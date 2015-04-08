# The public interface for bparse takes the following arguments:
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


# Single cpu version:
function bparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer, feats::Fvec, net::Net, 
                           nbeam::Integer, nbatch::Integer=1; xy::Bool=false)
    pa = map(s->pt(wcnt(s), ndeps), c)
    if !xy
        bparse(pa, c, ndeps, feats, net, nbeam, nbatch)
    else
        xtype = wtype(c[1])
        x = Array(xtype, xsize(pa[1], c, feats))
        y = zeros(xtype, ysize(pa[1], c))
        bparse(pa, c, ndeps, feats, net, nbeam, nbatch, x, y)
    end
    (xy ? (pa, x, y) : pa)
end

# Multi cpu version:
function bparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer, feats::Fvec, net::Net, 
                           nbeam::Integer, nbatch::Integer, ncpu::Integer; xy::Bool=false)
    ncpu == 1 && return bparse(pt, c, ndeps, feats, net, nbeam, nbatch; xy=xy)
    @date Main.resetworkers(ncpu)
    sa = distribute(c)                                  # distributed sentence array
    pa = map(s->pt(wcnt(s), ndeps), sa)                 # distributed parser array
    net = testnet(net)                                  # host copy of net for sharing
    if !xy
        @sync for p in procs(sa)
            @spawnat p bparse(localpart(pa), localpart(sa), ndeps, feats, copy(net,:gpu), nbeam, nbatch)
        end
    else
        xtype = wtype(c[1])
        x = SharedArray(xtype, xsize(pa[1],c,feats))    # shared x array
        y = SharedArray(xtype, ysize(pa[1],c))          # shared y array
        fill!(y, zero(xtype))
        nx = zeros(Int, length(c))                      # 1+nx[i] is the starting x column for i'th sentence
        p1 = pt(1,ndeps)
        for i=1:length(c)-1
            nx[i+1] = nx[i] + nmoves(p1, c[i])
        end
        @sync for p in procs(sa)
            @spawnat p bparse(localpart(pa), localpart(sa), ndeps, feats, copy(net,:gpu), nbeam, nbatch, x, y, nx[localindexes(sa)[1][1]])
        end
    end
    pa = convert(Vector{pt}, pa)
    @date Main.rmworkers()
    (xy ? (pa, sdata(x), sdata(y)) : pa)
end


# Data structure for beam search:
type Beam cmove; cost; cparser; cscore; csorted; nbeam; parser; parser2; pscore; pscore2; sentence; 
    function Beam(p::Parser, s::Sentence, ndeps::Integer, feats::Fvec, net::Net, nbeam::Integer)
        @assert (isdefined(net[end],:f) && net[end].f == KUnet.logp) "Need logp final layer for beam parser"
        b = new()
        nword = wcnt(s)
        ftype = wtype(s)
        itype = typeof(nbeam)
        ptype = typeof(p)
        ncand = nbeam * p.nmove
        b.cmove = Array(Move, ncand)
        b.cost = Array(Position, p.nmove, nbeam)
        b.cparser = Array(itype, ncand)
        b.cscore = Array(ftype, ncand)
        b.csorted = Array(itype, ncand)
        b.nbeam = 1
        b.parser  = [ptype(nword,ndeps) for i=1:nbeam]
        b.parser2 = [ptype(nword,ndeps) for i=1:nbeam]
        b.pscore  = Array(ftype, nbeam)
        b.pscore2 = Array(ftype, nbeam)
        b.pscore[1] = zero(ftype)
        b.sentence = s
        return b
    end
end

# Here is the workhorse:
function bparse{T<:Parser}(p::Vector{T}, corpus::Corpus, ndeps::Integer, feats::Fvec, net::Net, nbeam::Integer, nbatch::Integer,
                           x::AbstractArray=[], y::AbstractArray=[], nx::Integer=0)
    (nbatch == 0 || nbatch > length(corpus)) && (nbatch = length(corpus))
    ftype = wtype(corpus[1])
    frows = flen(p[1], corpus[1], feats)
    fcols = nbeam * nbatch
    f = Array(ftype, frows, fcols)
    score = Array(ftype, p[1].nmove, fcols)

    for s1=1:nbatch:length(corpus)                                              # processing corpus[s1:s2]
        s2=min(length(corpus), s1+nbatch-1)                                     
        batch = [Beam(p[i], corpus[i], ndeps, feats, net, nbeam) for i=s1:s2]   # initialize beam for each sentence
        while true                                              
            nf = 0                                                              # f[1:nf]: feature vectors for the whole batch s1:s2
            for b in batch                                                      # b is the beam (multiple parser states) for one sentence in s1:s2
                anyvalidmoves(b.parser[1]) || continue                          # assuming all parsers for a sentence finish at the same time
                (cmin,jmin,fmin) = (Pinf,0,0)                                   # mincost, its move, its nf index
                for i=1:b.nbeam                                                 # b.parser[i] is a parser state on b
                    features(b.parser[i], b.sentence, feats, f, (nf+=1))	# f[:,nf] is the feature vector for b.parser[i]
                    cost = movecosts(b.parser[i], b.sentence.head, 
                                     b.sentence.deprel, sub(b.cost,:,i))        # cost[j]=b.cost[j,i] is the cost of move j from parser i
                    cmini,jmini = findmin(cost)                                 # we only want to copy the mincost f column into x for each sentence
                    @assert (cmini < Pinf)                                      # we should have valid moves
                    (cmini < cmin) && ((cmin,jmin,fmin) = (cmini,jmini,nf))
                end # for i=1:b.nbeam
                @assert (cmin < Pinf)
                if !isempty(x)                                                  # if asked for x, copy mincost column from f
                    copy!(x, nx*frows+1, f, (fmin-1)*frows+1, frows)
                    nx += 1
                    y[jmin,nx]=one(eltype(y))                                   # and set mincost y move to 1
                end
            end # for b in batch (1)
            nf == 0 && break                                                    # no more valid moves for any sentence
            predict(net, sub(f,:,1:nf), sub(score,:,1:nf))                      # scores in score[1:nf]
            nf1 = nf; nf = 0                                                    # nf will count 1..nf1 during second pass
            for b in batch                                                      # collect candidates in second pass
                anyvalidmoves(b.parser[1]) || continue
                nc = 0                                                          # nc is number of candidates for sentence beam b
                for i=1:b.nbeam
                    @assert any(sub(b.cost,:,i) .< Pinf)
                    nf += 1                                                     # score[j,nf] should be the score for sentence b, parser i, move j
                    for j=1:size(b.cost,1)                                      # b.cost[j,i] is the cost of parser i, move j
                        b.cost[j,i] == Pinf && continue
                        nc += 1
                        b.cparser[nc] = i                                       # b.cparser[nc] is the index of the nc'th candidate parser
                        b.cmove[nc] = j                                         # b.cmove[nc] is the move to be made from b.cparser[nc]
                        b.cscore[nc] = b.pscore[i] + score[j,nf]                # b.cscore[nc] is the score for b.cparser[nc]+b.cmove[nc]
                    end # for j=1:size(b.cost,1)
                end # for i=1:b.nbeam (1)
                @assert (nc > 0) "No candidates found"
                sortpermx(sub(b.csorted, 1:nc), sub(b.cscore, 1:nc); rev=true)
                b.nbeam = min(nc,nbeam)                                         # b.nbeam is now the new beam size
                for new=1:b.nbeam                                               # new is the index of the new parser
                    idx=b.csorted[new]                                          # idx is the index of the new'th best candidate
                    old=b.cparser[idx]                                          # old is the index of the parent parser
                    copy!(b.parser2[new], b.parser[old])                        # copy the old parser
                    move!(b.parser2[new], b.cmove[idx])                         # move the new parser
                    b.pscore2[new] = b.cscore[idx]                              # b.pscore2[new] cumulative score for b.parser2[new]
                end # for new=1:b.nbeam (2)
                b.parser,b.parser2 = b.parser2,b.parser                         # we swap parsers and scores
                b.pscore,b.pscore2 = b.pscore2,b.pscore                         # for next round
            end # for b in batch (2)
            @assert (nf == nf1) "$nf != $nf1"
        end # while true
        for s=s1:s2
            copy!(p[s], batch[s-s1+1].parser[1])                                # copy the best parses to output
        end
    end # for s1=1:nbatch:length(corpus)
end # function bparse
