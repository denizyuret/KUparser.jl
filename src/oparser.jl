"""

    oparse(pt::Type{T<:Parser}, c::Union{Corpus,Sentence},
           ndeps:Integer, feats=nothing; usepmap=false)

The oracle parser executes the moves that will lead to the best (least
cost) parse.  It returns the final parse(s) and, if `feats` is
specified, the input features and correct moves for training.

## Arguments:
* pt::ParserType: ArcHybridR1, ArcEagerR1, ArcHybrid13, ArcEager13
* s::Sentence or c::Corpus: input sentence(s), a single parser is returned for s, a vector of parsers for c
* ndeps::Integer: number of dependency types
* feats::Fvec: (optional) specification of features, a (p,x,y) tuple returned if specified, only p if not
* usepmap::Bool: (optional) performs parallel processing

"""
function oparse{T<:Parser}(pt::Type{T}, s::Sentence, feats=nothing; o...)
    p = pt(s)
    op_work(p, s, feats)
end

function oparse{T<:Parser}(pt::Type{T}, c::Corpus, feats=nothing; usepmap::Bool=false)
    if usepmap
        op_pmap(pt, c, feats)
    else
        # op_main returns x,y in vectors for pcat
        p = op_main(pt, c, feats)
        if isa(p[1],Vector); (p[1],p[2][1],p[3][1]); else; p; end
    end
end

function op_pmap{T<:Parser}(pt::Type{T}, c::Corpus, feats)
    d = distribute(c)           # TODO: test what happens to common s.vocab
    p = pmap(procs(d)) do x
        op_main(pt, localpart(d), feats)
    end
    return pcat(p)
end

function op_main{T<:Parser}(pt::Type{T}, c::Corpus, feats)
    pa = map(pt, c)
    op_work(pa, c, feats)
end

# With no feats specified, just parse the sentences

function op_work{T<:Parser}(pa::Vector{T}, c::Corpus, feats::Void=nothing)
    for i=1:length(c)
        op_work(pa[i], c[i], feats)
    end
    return pa
end

function op_work(p::Parser, s::Sentence, feats::Void=nothing)
    c = Array(Cost, p.nmove)
    totalcost = 0
    while anyvalidmoves(p)
        movecosts(p, s.head, s.deprel, c)
        (bestcost,bestmove) = findmin(c)
        totalcost += bestcost
        move!(p, bestmove)
    end
    if totalcost != truecost(p,s); warn("Cost mismatch"); end
    return p
end

# With feats specified, parse the sentences and compute the features for training

function op_work{T<:Parser}(pa::Vector{T}, c::Corpus, feats::Fvec, x=nothing, y=nothing, nx=0)
    (x,y) = initoparse(pa,c,feats,x,y,nx)
    for i=1:length(c)
        op_work(pa[i], c[i], feats, x, y, nx)
        nx += nmoves(pa[i], c[i])
    end
    return (pa,typeof(x)[x],typeof(y)[y])  # must return tuple of three vectors for pcat
end

function op_work(p::Parser, s::Sentence, feats::Fvec, x=nothing, y=nothing, nx=0)
    (x,y) = initoparse(p,s,feats,x,y,nx)
    c = Array(Cost, p.nmove)
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
    if nx0 + nmoves(p,s) != nx; error(); end
    if totalcost != truecost(p,s); error("Cost mismatch"); end
    return (p,x,y)
end

function initoparse(p, s, f, x, y, n)
    if f == nothing; error(); end
    (xrows,xcols) = xsize(p,s,f)
    (yrows,ycols) = ysize(p,s)
    if xcols != ycols; error("Bad size"); end
    ytype = wtype(s)
    xtype = (if isa(f,DFvec); wtype(s); elseif isa(f,SFvec); SFtype; else; error(); end)
    mincols = xcols + n
    if y==nothing; y = zeros(ytype, yrows, mincols); end
    if x == nothing; x = Array(xtype, xrows, mincols); end
    if !(isa(y,AbstractMatrix{ytype}) && (size(y,1)==yrows) && (size(y,2)>=mincols)); error("Bad y"); end
    if !(isa(x,AbstractMatrix{xtype}) && (size(x,1)==xrows) && (size(x,2)>=mincols)); error("Bad x"); end
    return (x,y)
end

function pcat(p)
    if isa(p[1], Vector)
        q = similar(p[1], 0)
        for pi in p
            append!(q, pi)
        end
        return q
    else
        pp = similar(p[1][1],0)
        xx = similar(p[1][2],0)
        yy = similar(p[1][3],0)
        for i=1:length(p)
            append!(pp, p[i][1])
            append!(xx, p[i][2])
            append!(yy, p[i][3])
        end
        return (pp, xx, yy)
    end
end

### DEAD CODE

# function oparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer, feats::Fvec)
#     pa = map(s->pt(wcnt(s), ndeps), c)
#     oparse(pa, c, ndeps, feats)
# end

# function oparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer)
#     pa = map(s->pt(wcnt(s), ndeps), c)
#     oparse(pa, c, ndeps)
# end

# function oparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer, ncpu::Integer)
#     @date Main.resetworkers(ncpu)
#     sa = distribute(c)                                  # distributed sentence array
#     pa = map(s->pt(wcnt(s), ndeps), sa)                 # distributed parser array
#     @sync for p in procs(sa)
#         @spawnat p oparse(localpart(pa), localpart(sa), ndeps)
#     end
#     pa = convert(Vector{pt}, pa)
#     @date Main.rmworkers()
#     return pa
# end

# function oparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer, ncpu::Integer, feats::DFvec)
#     # Sparse fvec not supported yet, we need SharedSparseArray
#     @date Main.resetworkers(ncpu)
#     sa = distribute(c)                                  # distributed sentence array
#     pa = map(s->pt(wcnt(s), ndeps), sa)                 # distributed parser array
#     xtype = wtype(c[1])
#     x = SharedArray(xtype, xsize(pa[1],c,feats))    # shared x array
#     y = SharedArray(xtype, ysize(pa[1],c))          # shared y array
#     fill!(y, zero(xtype))
#     nx = zeros(Int, length(c))                      # 1+nx[i] is the starting x column for i'th sentence
#     p1 = pt(1,ndeps)
#     for i=1:length(c)-1
#         nx[i+1] = nx[i] + nmoves(p1, c[i])
#     end
#     @sync for p in procs(sa)
#         @spawnat p oparse(localpart(pa), localpart(sa), ndeps, feats, x, y, nx[localindexes(sa)[1][1]])
#     end
#     pa = convert(Vector{pt}, pa)
#     @date Main.rmworkers()
#     return (pa, sdata(x), sdata(y))
# end

# function oparse{T<:Parser}(pt::Type{T}, s::Sentence, ndeps::Integer)
#     p = pt(wcnt(s), ndeps)
#     oparse(p, s, ndeps)
# end

# function oparse{T<:Parser}(pt::Type{T}, s::Sentence, ndeps::Integer, feats::Fvec)
#     p = pt(wcnt(s), ndeps)
#     oparse(p, s, ndeps, feats)
# end

# function oparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer; usepmap::Bool=false)
#     if usepmap
#         d = distribute(c)
#         p = pmap(procs(d)) do x
#             oparse(pt, localpart(d), ndeps)
#         end
#         pcat(p)
#     else
#         pa = map(s->pt(wcnt(s), ndeps), c)
#         oparse(pa, c, ndeps)
#     end
# end

# function oparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer, feats::DFvec; usepmap::Bool=false)
#     if usepmap
#         d = distribute(c)
#         p = pmap(procs(d)) do x
#             oparse(pt, localpart(d), ndeps, feats)
#         end
#         pcat(p)
#     else
#         pa = map(s->pt(wcnt(s), ndeps), c)
#         oparse(pa, c, ndeps, feats)
#     end
# end

# function op_work(p::Parser, s::Sentence, ndeps::Integer, feats::SFvec, x=nothing, y=nothing, nx=0)
#     (x,y) = initoparse(p,s,ndeps,feats,x,y,nx)
#     c = Array(Position, p.nmove)
#     totalcost = 0; nx0 = nx
#     while anyvalidmoves(p)
#         movecosts(p, s.head, s.deprel, c)
#         (bestcost,bestmove) = findmin(c)
#         totalcost += bestcost
#         # We let features directly write in x.rowval[x.colptr[nx]:x.colptr[nx+1]-1]
#         features(p, s, feats, x, (nx+=1))
#         # We do not need this check any more, x.m is set to typemax
#         # features() sorts rowval so the max can be found at the end
#         # maxrow = x.rowval[x.colptr[nx+1]-1]
#         # If max rowval exceeds matrix height, we update the height
#         # maxrow > x.m && (x.m = maxrow)
#         y[:, nx] = zero(eltype(y))
#         y[bestmove, nx] = one(eltype(y))
#         move!(p, bestmove)
#     end
#     if nx0 + nmoves(p,s) != nx; error(); end
#     if totalcost != truecost(p,s); error("Cost mismatch"); end
#     return (p,x,y)
# end

