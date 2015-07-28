# The public interface for gparse takes the following arguments:
#
# pt::ParserType: ArcHybridR1, ArcEagerR1, ArcHybrid13, ArcEager13
# c::Corpus: array of input sentences
# ndeps::Integer: number of dependency types
# feats::Fvec: specification of features
# net::Net: model used for move prediction
# nbatch::Integer: (optional) parse sentences in batches for efficiency
# ncpu::Integer: (optional) perform parallel processing
# xy::Bool: (keyword) return (p,x,y) tuple for training, by default only parsers returned.

function gparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer, feats::Fvec, net::Net, 
                           nbatch::Integer=1, ncpu::Integer=1; xy::Bool=false)
    if ncpu > 1
        @assert ncpu==nworkers() "ncpu must be 1 or nworkers()"
        d = distribute(c)
        net = testnet(net)
        pmap(procs(d)) do x
            gparse(pt, localpart(d), ndeps, feats, gpucopy(net), nbatch, 1; xy=xy)
        end
    else
        pa = map(s->pt(wcnt(s), ndeps), c)
        if xy
            xtype = wtype(c[1])
            x = Array(xtype, xsize(pa[1], c, feats))
            y = zeros(xtype, ysize(pa[1], c))
            gparse(pa, c, ndeps, feats, net, nbatch, x, y)
            return (pa, x, y)
        else
            gparse(pa, c, ndeps, feats, net, nbatch)
            return pa
        end
    end
end

# Here is the workhorse:
function gparse{T<:Parser}(p::Vector{T}, c::Corpus, ndeps::Integer, feats::Fvec, net::Net, nbatch::Integer,
                           x::AbstractArray=[], y::AbstractArray=[], nx::Integer=0)
    (nbatch == 0 || nbatch > length(c)) && (nbatch = length(c))
    isempty(x) && (x=Array(wtype(c[1]), flen(p[1],c[1],feats), nbatch))
    score = Array(wtype(c[1]), p[1].nmove, nbatch)
    cost = Array(Position, p[1].nmove)

    for s1 = 1:nbatch:length(c)
        s2 = min(length(c), s1+nbatch-1)
        while true
            nx0 = nx            
            for s=s1:s2
                anyvalidmoves(p[s]) || continue;
                features(p[s], c[s], feats, x, (nx+=1))
            end
            (nx == nx0) && break
            nx1 = nx; nx = nx0; 
            predict(net, sub(sdata(x), :, nx0+1:nx1), sub(score, :, 1:(nx1-nx0)))
            for s=s1:s2
                anyvalidmoves(p[s]) || continue; nx += 1
                movecosts(p[s], c[s].head, c[s].deprel, cost)
                isempty(y) || (y[:,nx]=zero(eltype(y)); y[indmin(cost),nx]=one(eltype(y)))
                move!(p[s], maxscoremove(score, cost, nx-nx0))
            end
            @assert nx == nx1
            isempty(y) && (nx=0) # reuse x array if we are not returning xy
        end # while true
    end # for s1 = 1:nbatch:length(c)
    return nx
end

function maxscoremove(score, cost=[], col=1)
    (smax,imax) = (-Inf,0)
    for i=1:size(score,1)
        !isempty(cost) && (cost[i]==typemax(Cost)) && continue
        ((s = score[i,col]) > smax) && ((smax,imax)=(s,i))
    end
    return imax
end


# function gparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer, feats::Fvec, net::Net, 
#                            nbatch::Integer, ncpu::Integer; xy::Bool=false)
#     d = distribute(c)
#     net = testnet(net)
#     pmap(procs(d)) do x
#         gparse(pt, localpart(d), ndeps, feats, gpucopy(net), nbatch; xy=xy)
#     end
# end

# function gparse{T<:Parser}(pt::Type{T}, c::Corpus, ndeps::Integer, feats::Fvec, net::Net, 
#                            nbatch::Integer, ncpu::Integer; xy::Bool=false)
#     @date Main.resetworkers(ncpu)
#     sa = distribute(c)                                  # distributed sentence array
#     pa = map(s->pt(wcnt(s), ndeps), sa)                 # distributed parser array
#     net = testnet(net)                                  # host copy of net for sharing
#     if !xy
#         @sync for p in procs(sa)
#             @spawnat p gparse(localpart(pa), localpart(sa), ndeps, feats, copy(net,:gpu), nbatch)
#         end
#     else
#         xtype = wtype(c[1])
#         x = SharedArray(xtype, xsize(pa[1],c,feats))    # shared x array
#         y = SharedArray(xtype, ysize(pa[1],c))          # shared y array
#         fill!(y, zero(xtype))
#         nx = zeros(Int, length(c))                      # 1+nx[i] is the starting x column for i'th sentence
#         p1 = pt(1,ndeps)
#         for i=1:length(c)-1
#             nx[i+1] = nx[i] + nmoves(p1, c[i])
#         end
#         @sync for p in procs(sa)
#             @spawnat p gparse(localpart(pa), localpart(sa), ndeps, feats, copy(net,:gpu), nbatch, x, y, nx[localindexes(sa)[1][1]])
#         end
#     end
#     pa = convert(Vector{pt}, pa)
#     @date Main.rmworkers()
#     (xy ? (pa, sdata(x), sdata(y)) : pa)
# end
