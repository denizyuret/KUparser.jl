# The oracle parser executes the moves that will lead to the best parse.
# The public interface for oparse takes the following arguments:
#
# pt::ParserType: parser type: :ArcHybrid, :ArcEager, :ArcHybrid13, :ArcEager13
# s::Sentence or c::Corpus: input sentence(s), a single parser is returned for s, a vector of parsers for c
# f::Features: specification of features
# ndeps::Integer: number of dependency types
# ncpu::Integer: (optional) performs parallel processing
# xy::Bool: (optional, default=false) causes training data to be returned in a tuple (p, x, y)

function oparse{T<:Parser}(pt::Type{T}, s::Sentence, f::Features, ndeps::Integer, xy::Bool=false)
    p = pt(wcnt(s), ndeps)
    oparse(p, s, f, ndeps, xy)
end

function oparse{T<:Parser}(pt::Type{T}, c::Corpus, f::Features, ndeps::Integer, xy::Bool=false)
    pa = map(s->pt(wcnt(s), ndeps), c)
    oparse(pa, c, f, ndeps, xy)
end

function oparse{T<:Parser}(pt::Type{T}, c::Corpus, f::Features, ndeps::Integer, ncpu::Integer, xy::Bool=false)
    @date Main.resetworkers(ncpu)
    sa = distribute(c)                                  # distributed sentence array
    pa = map(s->pt(wcnt(s), ndeps), sa)                 # distributed parser array
    if xy
        xtype = wtype(c[1])
        x = SharedArray(xtype, xsize(pa[1],c,f))        # shared x array
        y = SharedArray(xtype, ysize(pa[1],c))          # shared y array
        fill!(y, zero(xtype))
        nx = zeros(Int, length(c))                      # 1+nx[i] is the starting x column for i'th sentence
        p1 = pt(1,ndeps)
        for i=1:length(c)-1
            nx[i+1] = nx[i] + nmoves(p1, c[i])
        end
        @sync for p in procs(sa)
            @spawnat p oparse(localpart(pa), localpart(sa), f, ndeps, true, x, y, nx[localindexes(sa)[1][1]])
        end
    else
        @sync for p in procs(sa)
            @spawnat p oparse(localpart(pa), localpart(sa), f, ndeps)
        end
    end
    pa = convert(Vector{pt}, pa)
    @date Main.rmworkers()
    return (xy ? (pa, sdata(x), sdata(y)) : pa)
end

function oparse(p::Parser, s::Sentence, f::Features, ndeps::Integer, 
                xy::Bool=false, 
                x::AbstractArray=(xy ? Array(wtype(s),xsize(p,s,f)): []), 
                y::AbstractArray=(xy ? zeros(wtype(s),ysize(p,s))  : []),
                nx::Integer=0)
    c = Array(Position, p.nmove)
    totalcost = 0; nx0 = nx
    while anyvalidmoves(p)
        movecosts(p, s.head, s.deprel, c)
        (bestcost,bestmove) = findmin(c)
        totalcost += bestcost
        if xy
            features(p, s, f, x, (nx+=1))
            y[bestmove, nx] = one(eltype(y))
        end
        move!(p, bestmove)
    end
    @assert (!xy || (nx0 + nmoves(p,s) == nx))
    @assert (totalcost == truecost(p,s))
    return (xy ? (p,x,y) : p)
end

function oparse{T<:Parser}(pa::Vector{T}, c::Corpus, f::Features, ndeps::Integer, 
                           xy::Bool=false, 
                           x::AbstractArray=(xy ? Array(wtype(c[1]),xsize(pa[1],c,f)): []), 
                           y::AbstractArray=(xy ? zeros(wtype(c[1]),ysize(pa[1],c))  : []),
                           nx::Integer=0)
    for i=1:length(c)
        oparse(pa[i], c[i], f, ndeps, xy, x, y, nx)
        xy && (nx += nmoves(pa[i], c[i]))
    end
    return (xy ? (pa,x,y) : pa)
end

function truecost(p::Parser, s::Sentence)
    cost = 0
    for i=1:p.nword
        ((p.head[i] != s.head[i]) || (p.deprel[i] != s.deprel[i])) && (cost += 1)
    end
    return cost
end

