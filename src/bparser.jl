type Beam nbeam; sentence; parser; parser2; pscore; pscore2; cost; score; cparser; cmove; cscore; csorted; mincost; Beam()=new(); end

function bparse(sentence::Sentence, net::Net, feats::Features, nbeam::Integer)
    b = Beam(sentence, net, feats, nbeam)                       # b.parser, b.pscore: candidate parsers and their scores
    (x,y) = initxy(sentence, net, feats, nbeam)                 # x:feature vectors, y:mincost moves
    nx = 0                                                      # nx: number of columns filled in x and y
    while true
        for i=1:b.nbeam                                         # b.nbeam: number of parsers on the beam
            cost(b.parser[i], sentence.head, sub(b.cost,:,i))   # b.cost[j,i] is the cost of j'th move for i'th parser
        end
        all(sub(b.cost,:,1:b.nbeam) .== Pinf) && break
        for i=1:b.nbeam
            features(b.parser[i], sentence, feats, sub(x,:,nx+i)) # x[:,nx+i] is the feature vector for the i'th parser
            y[indmin(sub(b.cost,:,i)),nx+i] = one(eltype(y))      # y[j,nx+i]=1 if j is the mincost move for i'th parser
        end
        KUnet.predict(net, sub(x,:,nx+1:nx+b.nbeam), b.score)   # b.score[j,i] is the score for the j'th move of i'th parser
        nx += b.nbeam
        nc = 0                                                  # nc is the number of new candidates
        for i=1:b.nbeam
            for j=1:size(b.cost,1)
                b.cost[j,i] == Pinf && continue
                nc += 1
                b.cparser[nc] = i                               # b.cparser[c] is the index of the c'th candidate parser
                b.cmove[nc] = j                                 # b.cmove[c] is the move to be made from b.cparser[c]
                b.cscore[nc] = b.pscore[i] + b.score[j,i]       # b.cscore[c] is the score for b.cparser[c]+b.cmove[c]
            end                                                 
        end
        sortperm!(sub(b.csorted, 1:nc), sub(b.cscore, 1:nc); rev=true)
        b.nbeam = min(nc,nbeam)                                 # b.nbeam is now the new beam size
        for i=1:b.nbeam                                         # i is the index of the new parser
            c=b.csorted[i]                                      # c is the index of the candidate
            p=b.cparser[c]                                      # p is the index of the old parser
            copy!(b.parser2[i], b.parser[p])                    
            move!(b.parser2[i], b.cmove[c])                     # b.parser2[i] = b.parser[p] + b.cmove[c]
            b.pscore2[i] = b.cscore[c]                          # b.pscore2[i] cumulative score for new b.parser2[i]
        end
        b.parser,b.parser2 = b.parser2,b.parser                 # we swap parsers and scores
        b.pscore,b.pscore2 = b.pscore2,b.pscore                 # for next round
    end
    (b.parser[1].head, sub(x,:,1:nx), sub(y,:,1:nx))
end

function initxy(sentence::Sentence, net::Net, feats::Features, nbeam::Integer)
    nmove = 2 * (wcnt(sentence) - 1)
    xcols = nmove * nbeam
    xrows = flen(wdim(sentence), feats)    
    xtype = eltype(net[1].w)
    yrows = ArcHybrid(1).nmove
    x = Array(xtype, xrows, xcols)
    y = zeros(xtype, yrows, xcols)
    return (x,y)
end

function bparse(c::Corpus, n::Net, f::Features, b::Integer)
    p = map(s->bparse(s,n,f,b), c)
    h = map(z->z[1], p)
    x = hcat(map(z->z[2], p)...)
    y = hcat(map(z->z[3], p)...)
    (h, x, y)
end

function Beam(sentence::Sentence, net::Net, feats::Features, nbeam::Integer)
    @assert (isdefined(net[end],:f) && net[end].f == KUnet.logp) "Need logp final layer"
    b = Beam()
    nword = wcnt(sentence)
    nmove = ArcHybrid(1).nmove
    ncand = nbeam * nmove
    itype = typeof(nbeam)
    ftype = eltype(net[1].w)
    fdims = flen(wdim(sentence), feats)
    b.parser  = [ArcHybrid(nword) for i=1:nbeam]
    b.parser2 = [ArcHybrid(nword) for i=1:nbeam]
    b.pscore  = Array(ftype, nbeam)
    b.pscore2 = Array(ftype, nbeam)
    b.cost = Array(Pval, nmove, nbeam)
    b.score = Array(ftype, nmove, nbeam)
    b.cparser = Array(itype, ncand)
    b.cmove = Array(Move, ncand)
    b.cscore = Array(ftype, ncand)
    b.csorted = Array(itype, ncand)
    b.pscore[1] = zero(ftype)
    b.nbeam = 1
    b.sentence = sentence
    b.mincost = minimum(cost(b.parser[1], b.sentence.head))
    return b
end

function bparse(corpus::Corpus, net::Net, feats::Features, nbeam::Integer, nbatch::Integer)
    (nbatch == 0 || nbatch > length(corpus)) && (nbatch = length(corpus))
    (heads,f,x,y,score) = initbatch(corpus, net, feats, nbeam, nbatch)
    nx = 0                                                      # training data goes into x[1:nx], y[1:nx]
    for s1=1:nbatch:length(corpus)                              # processing corpus[s1:s2]
        s2=min(length(corpus), s1+nbatch-1)                      
        batch = [Beam(corpus[i], net, feats, nbeam) for i=s1:s2] # initialize beam for each sentence
        while true                                              
            nf = 0                                        # patterns go into f[1:nf]
            for b in batch                                      # b is the beam for one sentence
                xadded = false                                  # only add one training instance for each sentence each move
                for i=1:b.nbeam                                 # i is a parser state on b
                    c = cost(b.parser[i], b.sentence.head, sub(b.cost,:,i))
                    all(c .== Pinf) && continue                 # c[j]=b.cost[j,i] is the cost of move j from parser i
                    nf += 1                                     # f[:,nf] is the feature vector for parser i
                    features(b.parser[i], b.sentence, feats, sub(f,:,nf))
                    if (!xadded && minimum(c) == b.mincost)         # add first mincost move to training data
                        nx += 1
                        copy!(sub(x,:,nx), sub(f,:,nf))
                        y[indmin(c),nx] = one(eltype(y))
                        xadded = true
                    end
                end # for i=1:b.nbeam
            end # for b in batch (1)
            nf == 0 && break
            KUnet.predict(net, sub(f,:,1:nf), sub(score,:,1:nf)) # scores in score[1:nf]
            mf = 0                                               # mf will count 1..nf during second pass
            for b in batch                                       # collect candidates in second pass
                all(sub(b.cost,:,1:b.nbeam) .== Pinf) && continue
                ncand = 0                                       # ncand is number of candidates for sentence beam b
                for i=1:b.nbeam
                    all(sub(b.cost,:,i) .== Pinf) && continue
                    mf += 1                                     # score[j,mf] should be the score for sentence b, parser i, move j
                    for j=1:size(b.cost,1)
                        b.cost[j,i] == Pinf && continue
                        ncand += 1
                        b.cparser[ncand] = i                        # b.cparser[c] is the index of the c'th candidate parser
                        b.cmove[ncand] = j                          # b.cmove[c] is the move to be made from b.cparser[c]
                        b.cscore[ncand] = b.pscore[i] + score[j,mf] # b.cscore[c] is the score for b.cparser[c]+b.cmove[c]
                    end # for j=1:size(b.cost,1)
                end # for i=1:b.nbeam (1)
                @assert (ncand > 0) "No candidates found"
                sortperm!(sub(b.csorted, 1:ncand), sub(b.cscore, 1:ncand); rev=true)
                b.nbeam = min(ncand,nbeam)                      # b.nbeam is now the new beam size
                for i=1:b.nbeam                                 # i is the index of the new parser
                    c=b.csorted[i]                              # c is the index of the candidate
                    p=b.cparser[c]                              # p is the index of the old parser
                    copy!(b.parser2[i], b.parser[p])            # 
                    move!(b.parser2[i], b.cmove[c])             # b.parser2[i] = b.parser[p] + b.cmove[c]
                    b.pscore2[i] = b.cscore[c]                  # b.pscore2[i] cumulative score for new b.parser2[i]
                end # for i=1:b.nbeam (2)
                b.parser,b.parser2 = b.parser2,b.parser         # we swap parsers and scores
                b.pscore,b.pscore2 = b.pscore2,b.pscore         # for next round
            end # for b in batch (2)
            @assert (mf == nf) "$mf != $nf"
        end # while true
        for s=s1:s2
            heads[s] = batch[s-s1+1].parser[1].head
        end
    end # for s1=1:nbatch:length(corpus)
    return (heads, sub(x, :, 1:nx), sub(y, :, 1:nx))
end # function bparse


function initbatch(corpus::Corpus, net::Net, feats::Features, nbeam::Integer, nbatch::Integer)
    # f,score to be used locally for one beam/batch
    xtype = eltype(net[1].w)
    xcols = nbeam * nbatch
    xrows = flen(wdim(corpus[1]), feats)    
    yrows = ArcHybrid(1).nmove
    f = Array(xtype, xrows, xcols)
    score = Array(xtype, yrows, xcols)
    # x,y collects mincostpath for the whole corpus
    nsent = length(corpus)
    nword = sum(map(wcnt, corpus))
    nmove = 2 * (nword - nsent)
    x = Array(xtype, xrows, nmove)
    y = zeros(xtype, yrows, nmove)
    heads = Array(Pvec, nsent)
    return (heads,f,x,y,score)
end

function bparse(corpus::Corpus, net::Net, fmat::Features, nbeam::Integer, nbatch::Integer, ncpu::Integer)
    assert(nworkers() >= ncpu)
    d = distproc(corpus, workers()[1:ncpu])
    n = testnet(net)
    @everywhere gc()
    p = pmap(procs(d)) do x
        bparse(localpart(d), copy(n, :gpu), fmat, nbeam, nbatch)
    end
    # h = vcat(map(z->z[1], p)...)
    # x = hcat(map(z->z[2], p)...)
    # y = hcat(map(z->z[3], p)...)
    # (h, x, y)
end

function bparse1(corpus::Corpus, net::Net, fmat::Features, nbeam::Integer, ncpu::Integer)
    assert(nworkers() >= ncpu)
    d = distproc(corpus, workers()[1:ncpu])
    n = testnet(net)
    @everywhere gc()
    p = pmap(procs(d)) do x
        bparse(localpart(d), copy(n, :gpu), fmat, nbeam)
    end
    # h = vcat(map(z->z[1], p)...)
    # x = hcat(map(z->z[2], p)...)
    # y = hcat(map(z->z[3], p)...)
    # (h, x, y)
end

