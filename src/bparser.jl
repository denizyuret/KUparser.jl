type Bparser nbeam; ncand; sentence; parser; parser2; pscore; pscore2; cost; score; x; cparser; cmove; cscore; csorted; Bparser()=new(); end

function bparse(sentence::Sentence, net::Net, feats::Features, beam::Integer)
    b::Bparser = initbeam(sentence, net, feats, beam)         # b.parser, b.pscore: candidate parsers and their scores
    while true
        for i=1:b.nbeam                                         # b.nbeam: number of parsers on the beam
            cost(b.parser[i], sentence.head, sub(b.cost,:,i))   # b.cost[j,i] is the cost of j'th move for i'th parser
        end
        all(sub(b.cost,:,1:b.nbeam) .== Pinf) && break
        for i=1:b.nbeam
            features(b.parser[i], sentence, feats, sub(b.x,:,i)) # b.x[:,i] is the feature vector for the i'th parser
        end
        KUnet.predict(net, b.x, b.score)                        # b.score[j,i] is the score for the j'th move of i'th parser
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
        b.nbeam = min(nc,beam)                                  # b.nbeam is now the new beam size
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
    b.parser[1].head
end

bparse(c::Corpus, n::Net, f::Features, b::Integer)=map(s->bparse(s,n,f,b), c)

function bparse2(corpus::Corpus, net::Net, fmat::Features, beam::Integer, ncpu::Integer)
    assert(nworkers() >= ncpu)
    d = distproc(corpus, workers()[1:ncpu])
    n = copy(net, :cpu)
    @everywhere gc()
    @time p = pmap(procs(d)) do x
        bparse(localpart(d), copy(n, :gpu), fmat, beam)
    end
    vcat(p...)
end

function initbeam(sentence::Sentence, net::Net, feats::Features, beam::Integer)
    assert(isdefined(net[end],:f) && net[end].f == KUnet.logp)
    b = Bparser()
    nword = wcnt(sentence)
    nmove = ArcHybrid(1).nmove
    ncand = beam * nmove
    itype = typeof(beam)
    ftype = eltype(net[1].w)
    fdims = flen(wdim(sentence), feats)
    b.parser  = [ArcHybrid(nword) for i=1:beam]
    b.parser2 = [ArcHybrid(nword) for i=1:beam]
    b.pscore  = Array(ftype, beam)
    b.pscore2 = Array(ftype, beam)
    b.cost = Array(Pval, nmove, beam)
    b.x = Array(ftype, fdims, beam)
    b.score = Array(ftype, nmove, beam)
    b.cparser = Array(itype, ncand)
    b.cmove = Array(Move, ncand)
    b.cscore = Array(ftype, ncand)
    b.csorted = Array(itype, ncand)
    b.pscore[1] = zero(ftype)
    b.nbeam = 1
    b.sentence = sentence
    return b
end


function bparse(corpus::Corpus, net::Net, feats::Features, nbeam::Integer, nbatch::Integer)
    (heads,x,y,score) = initbatch(corpus, net, feats, nbeam, nbatch)
    (x1,x2) = (0,0)
    for s1=1:nbatch:length(corpus)
        s2=min(length(corpus), s1+nbatch-1)
        batch = [initbeam(corpus[i], net, feats, nbeam) for i=s1:s2]
        while true                                              # processing corpus[s1:s2]
            x1 = x2 + 1                                         # put all patterns in x[x1:x2]
            for b in batch                                      # b is the beam for one sentence
                for i=1:b.nbeam                                 # i is a parser state on b
                    c = cost(b.parser[i], b.sentence.head, sub(b.cost,:,i))
                    all(c .== Pinf) && continue                 # c[j]=b.cost[j,i] is the cost of move j from state i
                    x2 += 1
                    features(b.parser[i], b.sentence, feats, sub(x,:,x2))
                    y[indmin(c),x2] = one(eltype(y))
                end # for i=1:b.nbeam
            end # for b in batch (1)
            x2 < x1 && break
            KUnet.predict(net, sub(x,:,x1:x2), sub(score,:,1:x2-x1+1))
            x3 = 0                                              # scores in score[1:x3=x2-x1+1]
            for b in batch
                all(sub(b.cost,:,1:b.nbeam) .== Pinf) && continue
                ncand = 0
                for i=1:b.nbeam
                    all(sub(b.cost,:,i) .== Pinf) && continue
                    x3 += 1
                    for j=1:size(b.cost,1)
                        b.cost[j,i] == Pinf && continue
                        ncand += 1
                        b.cparser[ncand] = i                        # b.cparser[c] is the index of the c'th candidate parser
                        b.cmove[ncand] = j                          # b.cmove[c] is the move to be made from b.cparser[c]
                        b.cscore[ncand] = b.pscore[i] + score[j,x3] # b.cscore[c] is the score for b.cparser[c]+b.cmove[c]
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
            @assert (x3 == x2-x1+1) "$x3 != $x2-$x1+1"
        end # while true
        for s=s1:s2
            heads[s] = batch[s-s1+1].parser[1].head
        end
    end # for s1=1:nbatch:length(corpus)
    return heads
end # function bparse


function initbatch(corpus::Corpus, net::Net, feats::Features, nbeam::Integer, nbatch::Integer)
    nsent = length(corpus)
    nword = sum(map(wcnt, corpus))
    nmove = 2 * (nword - nsent)
    xcols = nmove * nbeam
    xrows = flen(wdim(corpus[1]), feats)    
    xtype = eltype(net[1].w)
    yrows = ArcHybrid(1).nmove
    x = Array(xtype, xrows, xcols)
    y = zeros(xtype, yrows, xcols)
    score = Array(xtype, yrows, nbeam * nbatch)
    heads = Array(Pvec, nsent)
    return (heads,x,y,score)
end
