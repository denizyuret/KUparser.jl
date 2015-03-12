type Bparser parser; parser2; pscore; pscore2; cost; score; x; y; cparser; cmove; cscore; csorted; Bparser()=new(); end

function bparse(sentence::Sentence, net::Net, feats::Features, beam::Integer)
    b::Bparser = initbparse(sentence, net, feats, beam)         # b.parser[1..np] candidate parser states on the beam
    np = 1                                                      # b.pscore[1..np] are their cumulative scores
    while true
        for i=1:np                                              
            cost(b.parser[i], sentence.head, sub(b.cost,:,i))   # b.cost[j,i] is the cost of j'th move for i'th parser
        end
        all(sub(b.cost,:,1:np) .== Pinf) && break
        shiftxy(b, np)
        for i=1:np
            b.y[indmin(sub(b.cost,:,i)), i] = one(eltype(b.y))   # b.y[j,i]=1 if parser i has mincost move j
            features(b.parser[i], sentence, feats, sub(b.x,:,i)) # b.x[:,i] is the feature vector for the i'th parser
        end
        KUnet.predict(net, b.x, b.score)                        # b.score[j,i] is the score for the j'th move of i'th parser
        nc = 0                                                  # nc is the number of new candidates
        for i=1:np
            for j=1:size(b.cost,1)
                b.cost[j,i] == Pinf && continue
                nc += 1
                b.cparser[nc] = i                               # b.cparser[c] is the index of the c'th candidate parser
                b.cmove[nc] = j                                 # b.cmove[c] is the move to be made from b.cparser[c]
                b.cscore[nc] = b.pscore[i] + b.score[j,i]       # b.cscore[c] is the score for b.cparser[c]+b.cmove[c]
            end                                                 
        end
        sortperm!(sub(b.csorted, 1:nc), sub(b.cscore, 1:nc); rev=true)
        np = min(nc,beam)                                       # np is the new beam size
        for i=1:np                                              # i is the index of the new parser
            c=b.csorted[i]                                      # c is the index of the candidate
            p=b.cparser[c]                                      # p is the index of the old parser
            copy!(b.parser2[i], b.parser[p])                    
            move!(b.parser2[i], b.cmove[c])                     # b.parser2[i] = b.parser[p] + b.cmove[c]
            b.pscore2[i] = b.cscore[c]                          # b.pscore2[i] cumulative score for new b.parser2[i]
        end
        b.parser,b.parser2 = b.parser2,b.parser                 # we swap parsers and scores
        b.pscore,b.pscore2 = b.pscore2,b.pscore                 # for next round
    end
    idx = isempty(b.x.indexes[2]) ? 0 : maximum(b.x.indexes[2])
    b.x = sub(b.x.parent,:,1:idx)
    b.y = sub(b.y.parent,:,1:idx)
    (b.parser[1].head, b.x, b.y)
end

function bparse(c::Corpus, n::Net, f::Features, b::Integer; args...)
    map(s->bparse(s,n,f,b;args...), c)
end

function shiftxy(b, np)
    idx = isempty(b.x.indexes[2]) ? 0 : maximum(b.x.indexes[2])
    b.x = sub(b.x.parent,:,idx+1:idx+np)
    b.y = sub(b.y.parent,:,idx+1:idx+np)
end

function initbparse(sentence::Sentence, net::Net, feats::Features, beam::Integer)
    net[end].f = KUnet.logp                                     # TODO: warn about this
    b = Bparser()
    nword = wcnt(sentence)
    nmove = ArcHybrid(1).nmove
    ncand = beam * nmove
    xsize = 2 * (nword - 1) * beam
    itype = typeof(beam)
    ftype = eltype(net[1].w)
    fdims = flen(wdim(sentence), feats)
    b.parser  = [ArcHybrid(nword) for i=1:beam]
    b.parser2 = [ArcHybrid(nword) for i=1:beam]
    b.pscore  = Array(ftype, beam)
    b.pscore2 = Array(ftype, beam)
    b.cost = Array(Pval, nmove, beam)
    b.x = sub(Array(ftype, fdims, xsize), :, 1:0)
    b.y = sub(zeros(ftype, nmove, xsize), :, 1:0)
    b.score = Array(ftype, nmove, beam)
    b.cparser = Array(itype, ncand)
    b.cmove = Array(Move, ncand)
    b.cscore = Array(ftype, ncand)
    b.csorted = Array(itype, ncand)
    b.pscore[1] = zero(ftype)
    return b
end

# suggest this change to julia:
import Base: sortperm!, Algorithm, Ordering, DEFAULT_UNSTABLE, Forward, Perm, ord
function sortperm!{I<:Integer}(x::AbstractVector{I}, v::AbstractVector; alg::Algorithm=DEFAULT_UNSTABLE,
                               lt::Function=isless, by::Function=identity, rev::Bool=false, order::Ordering=Forward,
                               initialized::Bool=false)
    length(x) != length(v) && throw(ArgumentError("Index vector must be the same length as the source vector."))
    !initialized && @inbounds for i = 1:length(v); x[i] = i; end
    sort!(x, alg, Perm(ord(lt,by,rev,order),v))
end

