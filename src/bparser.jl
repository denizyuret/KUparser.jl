type Beam nbeam; sentence; parser; parser2; pscore; pscore2; cost; score; cparser; cmove; cscore; csorted; mincost; Beam()=new(); end

function bparse(pt::ParserType, sent::Sentence, net::Net, feats::Features, ndeps::Integer, nbeam::Integer)
    b = Beam(pt, sent, net, feats, ndeps, nbeam)                # b.parser, b.pscore: candidate parsers and their scores
    (f,x,y) = initfxy(pt, sent, net, feats, ndeps, nbeam)       # f,x:feature vectors, y:mincost moves
    nx = 0                                                      # nx: number of columns filled in x and y
    while true
        anyvalidmoves(b.parser[1]) || break                     # assuming all parsers finish at the same time
        (cmin,jmin,fmin) = (Pinf,0,0)                           # mincost, its move, its nf index
        for i=1:b.nbeam                                         # b.nbeam: number of parsers on the beam
            c = movecosts(b.parser[i], sent.head, sent.deprel, sub(b.cost,:,i))   # b.cost[j,i] is the cost of j'th move for i'th parser
            cmini,jmini = findmin(c)
            @assert (cmini < Pinf)
            features(b.parser[i], sent, feats, sub(f,:,i))      # f[:,i] is the feature vector for the i'th parser
            (cmini < cmin) && ((cmin,jmin,fmin) = (cmini,jmini,i))
        end
        @assert (cmin < Pinf)
        copy!(sub(x,:,(nx+=1)),sub(f,:,fmin))                   # x: state vector on mincostpath
        y[jmin,nx]=one(eltype(y))                               # y: correct move on mincostpath
        predict(net, sub(f,:,1:b.nbeam), b.score)               # b.score[j,i] is the score for the j'th move of i'th parser
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
        sortpermx(sub(b.csorted, 1:nc), sub(b.cscore, 1:nc); rev=true)
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
    @assert nx == size(x,2)
    (b.parser[1], x, y)
end

function bparse(pt::ParserType, corpus::Corpus, net::Net, feats::Features, ndeps::Integer, nbeam::Integer)
    pxy = map(s->bparse(pt,s,net,feats,ndeps,nbeam), corpus)
    p = map(z->z[1], pxy)
    x = hcat(map(z->z[2], pxy)...)
    y = hcat(map(z->z[3], pxy)...)
    (p, x, y)
end

function bparse(pt::ParserType, corpus::Corpus, net::Net, feats::Features, ndeps::Integer, nbeam::Integer, nbatch::Integer)
    (nbatch == 0 || nbatch > length(corpus)) && (nbatch = length(corpus))
    (parsers,f,x,y,score) = initbatch(pt, corpus, net, feats, ndeps, nbeam, nbatch)
    nx = 0                                                      # training data goes into x[1:nx], y[1:nx]
    for s1=1:nbatch:length(corpus)                              # processing corpus[s1:s2]
        s2=min(length(corpus), s1+nbatch-1)                      
        batch = [Beam(pt, corpus[i], net, feats, ndeps, nbeam) for i=s1:s2] # initialize beam for each sentence
        while true                                              
            nf = 0                                              # patterns go into f[1:nf]
            for b in batch                                      # b is the beam for one sentence
                anyvalidmoves(b.parser[1]) || continue          # assuming all parsers for a sentence finish at the same time
                (cmin,jmin,fmin) = (Pinf,0,0)                   # mincost, its move, its nf index
                for i=1:b.nbeam                                 # b.parser[i] is a parser state on b
                    c = movecosts(b.parser[i], b.sentence.head, b.sentence.deprel, sub(b.cost,:,i)) # c[j]=b.cost[j,i] is the cost of move j from parser i
                    cmini,jmini = findmin(c)
                    @assert (cmini < Pinf)                      # we should have valid moves
                    features(b.parser[i], b.sentence, feats, sub(f,:,(nf+=1))) # f[:,nf] is the feature vector for parser i
                    (cmini < cmin) && ((cmin,jmin,fmin) = (cmini,jmini,nf))
                end # for i=1:b.nbeam
                @assert (cmin < Pinf)
                copy!(sub(x,:,(nx+=1)),sub(f,:,fmin))
                y[jmin,nx]=one(eltype(y))
            end # for b in batch (1)
            nf == 0 && break                                     # no more valid moves for any sentence
            predict(net, sub(f,:,1:nf), sub(score,:,1:nf))       # scores in score[1:nf]
            mf = 0                                               # mf will count 1..nf during second pass
            for b in batch                                       # collect candidates in second pass
                anyvalidmoves(b.parser[1]) || continue
                ncand = 0                                       # ncand is number of candidates for sentence beam b
                for i=1:b.nbeam
                    @assert any(sub(b.cost,:,i) .< Pinf)
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
                sortpermx(sub(b.csorted, 1:ncand), sub(b.cscore, 1:ncand); rev=true)
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
            parsers[s] = batch[s-s1+1].parser[1]
        end
    end # for s1=1:nbatch:length(corpus)
    @assert nx == size(x,2)
    return (parsers, x, y)
end # function bparse


function bparse(pt::ParserType, corpus::Corpus, net::Net, fmat::Features, ndeps::Integer, nbeam::Integer, nbatch::Integer, ncpu::Integer)
    @date Main.resetworkers(ncpu)
    d = distproc(corpus, workers()[1:ncpu])
    net = testnet(net)
    pxy = pmap(procs(d)) do x
        bparse(pt, localpart(d), copy(net, :gpu), fmat, ndeps, nbeam, nbatch)
    end
    @date Main.rmworkers()
    p = vcat(map(z->z[1], pxy)...)
    x = hcat(map(z->z[2], pxy)...)
    y = hcat(map(z->z[3], pxy)...)
    (p, x, y)
end

function bparse1(pt::ParserType, corpus::Corpus, net::Net, fmat::Features, ndeps::Integer, nbeam::Integer, ncpu::Integer)
    @date Main.resetworkers(ncpu)
    d = distproc(corpus, workers()[1:ncpu])
    net = testnet(net)
    pxy = pmap(procs(d)) do x
        bparse(pt, localpart(d), copy(net, :gpu), fmat, ndeps, nbeam)
    end
    @date Main.rmworkers()
    p = vcat(map(z->z[1], pxy)...)
    x = hcat(map(z->z[2], pxy)...)
    y = hcat(map(z->z[3], pxy)...)
    (p, x, y)
end

function initfxy(pt::ParserType, sent::Sentence, net::Net, feats::Features, ndeps::Integer, nbeam::Integer)
    p = Parser{pt}(1,ndeps)
    xcols = 2 * (wcnt(sent) - 1)
    fcols = nbeam
    xrows = flen(p, sent, feats)
    yrows = p.nmove
    xtype = eltype(net[1].w)
    x = Array(xtype, xrows, xcols)
    y = zeros(xtype, yrows, xcols)
    f = Array(xtype, xrows, fcols)
    return (f,x,y)
end

function initbatch(pt::ParserType, corpus::Corpus, net::Net, feats::Features, ndeps::Integer, nbeam::Integer, nbatch::Integer)
    # x,y collects mincostpath for the whole corpus
    p = Parser{pt}(1,ndeps)
    nsent = length(corpus)
    nword = sum(map(wcnt, corpus))
    xcols = 2 * (nword - nsent)
    xrows = flen(p, corpus[1], feats)    
    yrows = p.nmove
    xtype = eltype(net[1].w)
    x = Array(xtype, xrows, xcols)
    y = zeros(xtype, yrows, xcols)
    # f,score to be used locally for one beam/batch
    fcols = nbeam * nbatch
    f = Array(xtype, xrows, fcols)
    score = Array(xtype, yrows, fcols)
    parsers = Array(Parser{pt}, nsent)
    return (parsers,f,x,y,score)
end

function Beam(pt::ParserType, sent::Sentence, net::Net, feats::Features, ndeps::Integer, nbeam::Integer)
    @assert (isdefined(net[end],:f) && net[end].f == KUnet.logp) "Need logp final layer"
    b = Beam()
    nword = wcnt(sent)
    b.parser  = [Parser{pt}(nword,ndeps) for i=1:nbeam]
    b.parser2 = [Parser{pt}(nword,ndeps) for i=1:nbeam]
    p = b.parser[1]
    ncand = nbeam * p.nmove
    itype = typeof(nbeam)
    ftype = eltype(net[1].w)
    fdims = flen(p, sent, feats)
    b.pscore  = Array(ftype, nbeam)
    b.pscore2 = Array(ftype, nbeam)
    b.cost = Array(Position, p.nmove, nbeam)
    b.score = Array(ftype, p.nmove, nbeam)
    b.cparser = Array(itype, ncand)
    b.cmove = Array(Move, ncand)
    b.cscore = Array(ftype, ncand)
    b.csorted = Array(itype, ncand)
    b.pscore[1] = zero(ftype)
    b.nbeam = 1
    b.sentence = sent
    b.mincost = minimum(movecosts(b.parser[1], b.sentence.head, b.sentence.deprel))
    return b
end


