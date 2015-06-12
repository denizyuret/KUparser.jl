# The oracle parser executes the moves that will lead to the best parse.
# The public interface for oparse takes the following arguments:
#
# pt::ParserType: ArcHybridR1, ArcEagerR1, ArcHybrid13, ArcEager13
# s::Sentence or c::Corpus: input sentence(s), a single parser is returned for s, a vector of parsers for c
# ndeps::Integer: number of dependency types
# ncpu::Integer: (optional) performs parallel processing
# feats::Fvec: (optional) specification of features, a (p,x,y) tuple returned if specified, only p if not


function oparse{T<:Parser}(pt::Type{T}, s::Sentence, ndeps::Integer)
    p = pt(wcnt(s), ndeps)
    oparse(p, s, ndeps)
end

function oparse{T<:Parser}(pt::Type{T}, s::Sentence, ndeps::Integer, feats::Fvec)
    p = pt(wcnt(s), ndeps)
    oparse(p, s, ndeps, feats)
end

function oparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer)
    pa = map(s->pt(wcnt(s), ndeps), c)
    oparse(pa, c, ndeps)
end

function oparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer, feats::Fvec)
    pa = map(s->pt(wcnt(s), ndeps), c)
    oparse(pa, c, ndeps, feats)
end

function oparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer, ncpu::Integer)
    @date Main.resetworkers(ncpu)
    sa = distribute(c)                                  # distributed sentence array
    pa = map(s->pt(wcnt(s), ndeps), sa)                 # distributed parser array
    @sync for p in procs(sa)
        @spawnat p oparse(localpart(pa), localpart(sa), ndeps)
    end
    pa = convert(Vector{pt}, pa)
    @date Main.rmworkers()
    return pa
end

function oparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer, ncpu::Integer, feats::DFvec)
    # Sparse fvec not supported yet, we need SharedSparseArray
    @date Main.resetworkers(ncpu)
    sa = distribute(c)                                  # distributed sentence array
    pa = map(s->pt(wcnt(s), ndeps), sa)                 # distributed parser array
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
    pa = convert(Vector{pt}, pa)
    @date Main.rmworkers()
    return (pa, sdata(x), sdata(y))
end

function oparse{T<:Parser}(pa::Vector{T}, c::Corpus, ndeps::Integer)
    for i=1:length(c)
        oparse(pa[i], c[i], ndeps)
    end
    return pa
end

function oparse{T<:Parser}(pa::Vector{T}, c::Corpus, ndeps::Integer, feats::Fvec, x=nothing, y=nothing, nx=0)
    (x,y) = initoparse(pa,c,ndeps,feats,x,y,nx)
    for i=1:length(c)
        oparse(pa[i], c[i], ndeps, feats, x, y, nx)
        nx += nmoves(pa[i], c[i])
    end
    return (pa,x,y)
end

function oparse(p::Parser, s::Sentence, ndeps::Integer)
    c = Array(Position, p.nmove)
    totalcost = 0
    while anyvalidmoves(p)
        movecosts(p, s.head, s.deprel, c)
        (bestcost,bestmove) = findmin(c)
        totalcost += bestcost
        move!(p, bestmove)
    end
    @assert (totalcost == truecost(p,s))
    return p
end

function oparse(p::Parser, s::Sentence, ndeps::Integer, feats::Fvec, x=nothing, y=nothing, nx=0)
    (x,y) = initoparse(p,s,ndeps,feats,x,y,nx)
    c = Array(Position, p.nmove)
    totalcost = 0; nx0 = nx
    while anyvalidmoves(p)
        movecosts(p, s.head, s.deprel, c)
        (bestcost,bestmove) = findmin(c)
        totalcost += bestcost
        features(p, s, feats, x, (nx+=1))
        y[:, nx] = zero(eltype(y))
        y[bestmove, nx] = one(eltype(y))
        move!(p, bestmove)
    end
    @assert (nx0 + nmoves(p,s) == nx)
    @assert (totalcost == truecost(p,s))
    return (p,x,y)
end

function initoparse(p, s, d, f, x, y, n)
    @assert f != nothing
    (xrows,xcols) = xsize(p,s,f)
    (yrows,ycols) = ysize(p,s)
    @assert xcols == ycols
    mincols = xcols + n
    xtype = wtype(s)
    (y==nothing) && (y = zeros(xtype, yrows, mincols))
    @assert isa(y,AbstractMatrix{xtype}) && (size(y,1)==yrows) && (size(y,2)>=mincols)
    if isa(f, DFvec)
        (x==nothing) && (x = Array(xtype, xrows, mincols))
        @assert isa(x,AbstractMatrix{xtype}) && (size(x,1)==xrows) && (size(x,2)>=mincols)
    elseif isa(f, SFvec)
        (x==nothing) && (x = spzeros(xtype, xrows, mincols))
        @assert isa(x,SparseMatrixCSC{xtype}) && (size(x,1)==xrows) && (size(x,2)>=mincols)
    else
        error("Do not recognize fvec")
    end
    return (x,y)
end

