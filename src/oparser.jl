# The oracle parser executes the moves that will lead to the best parse.
# The public interface for oparse takes the following arguments:
#
# pt::ParserType: ArcHybridR1, ArcEagerR1, ArcHybrid13, ArcEager13
# s::Sentence or c::Corpus: input sentence(s), a single parser is returned for s, a vector of parsers for c
# ndeps::Integer: number of dependency types
# ncpu::Integer: (optional) performs parallel processing
# feats::Fvec: (optional) specification of features, a (p,x,y) tuple returned if specified, only p if not


function oparse{T<:Parser}(pt::Type{T}, s::Sentence, ndeps::Integer, feats::Fvec=Feature[])
    p = pt(wcnt(s), ndeps)
    oparse(p, s, ndeps, feats)
end

function oparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer, feats::Fvec=Feature[])
    pa = map(s->pt(wcnt(s), ndeps), c)
    oparse(pa, c, ndeps, feats)
end

function oparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer, ncpu::Integer, feats::Fvec=Feature[])
    @date Main.resetworkers(ncpu)
    sa = distribute(c)                                  # distributed sentence array
    pa = map(s->pt(wcnt(s), ndeps), sa)                 # distributed parser array
    if !isempty(feats)
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
            @spawnat p oparse(localpart(pa), localpart(sa), ndeps, feats, x, y, nx[localindexes(sa)[1][1]])
        end
    else
        @sync for p in procs(sa)
            @spawnat p oparse(localpart(pa), localpart(sa), ndeps, feats)
        end
    end
    pa = convert(Vector{pt}, pa)
    @date Main.rmworkers()
    return (isempty(feats) ? pa : (pa, sdata(x), sdata(y)))
end

function oparse(p::Parser, s::Sentence, ndeps::Integer, feats::Fvec=Feature[], 
                x::AbstractArray=(isempty(feats) ? [] : Array(wtype(s),xsize(p,s,feats))), 
                y::AbstractArray=(isempty(feats) ? [] : zeros(wtype(s),ysize(p,s))),
                nx::Integer=0)
    c = Array(Position, p.nmove)
    totalcost = 0; nx0 = nx
    while anyvalidmoves(p)
        movecosts(p, s.head, s.deprel, c)
        (bestcost,bestmove) = findmin(c)
        totalcost += bestcost
        if !isempty(feats)
            features(p, s, feats, x, (nx+=1))
            y[bestmove, nx] = one(eltype(y))
        end
        move!(p, bestmove)
    end
    @assert (isempty(feats) || (nx0 + nmoves(p,s) == nx))
    @assert (totalcost == truecost(p,s))
    return (isempty(feats) ? p : (p,x,y))
end

function oparse{T<:Parser}(pa::Vector{T}, c::Corpus, ndeps::Integer, feats::Fvec=Feature[], 
                           x::AbstractArray=(isempty(feats) ? [] : Array(wtype(c[1]),xsize(pa[1],c,feats))), 
                           y::AbstractArray=(isempty(feats) ? [] : zeros(wtype(c[1]),ysize(pa[1],c))),
                           nx::Integer=0)
    for i=1:length(c)
        oparse(pa[i], c[i], ndeps, feats, x, y, nx)
        isempty(feats) || (nx += nmoves(pa[i], c[i]))
    end
    return (isempty(feats) ? pa : (pa,x,y))
end

