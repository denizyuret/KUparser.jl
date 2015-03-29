# We define Parser{T} as a parametric type where T is :ArcHybrid,
# :ArcEager etc. to allow specialization of methods.  Thanks to
# julia-users members Toivo Henningsson and Simon Danisch for
# suggesting this design.


typealias ParserType Symbol
@compat typealias Position UInt8
@compat typealias DepRel UInt8
typealias Move Integer
typealias Pvec AbstractVector{Position}
typealias Dvec AbstractVector{DepRel}
typealias Pmat AbstractMatrix{Position}
const Pinf=typemax(Position)
Pzeros(n::Integer...)=zeros(Position, n...)
Dzeros(n::Integer...)=zeros(DepRel, n...)


type Parser{T}
    nword::Position       # number of words in sentence
    ndeps::DepRel         # number of dependency labels (excluding ROOT)
    nmove::Move           # number of possible moves (set by init!)
    wptr::Position        # index of first word in buffer
    sptr::Position        # index of last word (top) of stack
    stack::Pvec           # 1xn vector for stack of indices
    head::Pvec            # 1xn vector of heads
    deprel::Dvec          # 1xn vector of dependency labels
    lcnt::Pvec            # lcnt(h): number of left deps for h
    rcnt::Pvec            # rcnt(h): number of right deps for h
    ldep::Pmat            # nxn matrix for left dependents
    rdep::Pmat            # nxn matrix for right dependents
    
    function Parser(nword::Integer, ndeps::Integer)
        @assert (nword < typemax(Position)) "nword >= $(typemax(Position))"
        @assert (ndeps < typemax(DepRel)) "ndeps >= $(typemax(DepRel))"
        p = new(nword, ndeps, 0,               # nword, ndeps, nmove
                1, 0, Pzeros(nword),           # wptr, sptr, stack
                Pzeros(nword), Dzeros(nword),  # head, deprel
                Pzeros(nword), Pzeros(nword),  # lcnt, rcnt
                Pzeros(nword,nword), Pzeros(nword,nword)) # ldep, rdep
        init!(p)
        return p
    end
end # Parser


# A parser provides three functions: anyvalidmoves(), movecosts(), and move!():

anyvalidmoves(p::Parser)=(shiftok(p)||reduceok(p)||leftok(p)||rightok(p))

function movecosts(p::Parser, head::AbstractArray, deprel::AbstractArray, cost::Pvec=Array(Position,p.nmove))
    eagercosts(p,head,deprel,cost)
end

function move!(p::Parser, m::Move)
    @assert (1 <= m <= p.nmove) "Move $m is not supported"
    (m == shiftmove(p))  ? (@assert shiftok(p);  shift(p)) :
    in(m, rightmoves(p)) ? (@assert rightok(p);  right(p,label(p,m))) :
    in(m, leftmoves(p))  ? (@assert leftok(p);   left(p,label(p,m))) :
    (m == reducemove(p)) ? (@assert reduceok(p); reduce(p)) :
    error("Move $m is not supported")
end


# We provide default definitions for the ArcEager system.

# Moves are represented by integers 1..p.nmove
# Here they correspond to REDUCE,L1,R1,L2,R2,..,L[ndeps],R[ndeps],SHIFT

init!(p::Parser)=(p.nmove=(2+p.ndeps<<1);shift(p))
label(p::Parser,m::Move)=convert(DepRel,m>>1)
shiftmove(p::Parser)=p.nmove
reducemove(p::Parser)=1
leftmove(p::Parser,l::DepRel)=(l<<1)
rightmove(p::Parser,l::DepRel)=(1+l<<1)
leftmoves(p::Parser)=(2:2:(p.nmove-2))
rightmoves(p::Parser)=(3:2:(p.nmove-1))

# Specify what moves are valid by overriding these

shiftok(p::Parser)=(p.wptr <= p.nword)
rightok(p::Parser)=((p.wptr <= p.nword) && (p.sptr > 0))
leftok(p::Parser)=((p.wptr <= p.nword) && (p.sptr > 0) && !s0head(p))
reduceok(p::Parser)=((p.sptr > 0) && (s0head(p) || s0head0(p)))

s0head(p::Parser)=(p.head[p.stack[p.sptr]] != 0)
s0head0(p::Parser)=((p.wptr > p.nword) && (p.sptr > 1))

# Specify what moves actually do by overriding these

shift(p::Parser)=(p.stack[p.sptr+=1]=p.wptr; p.wptr+=1)
reduce(p::Parser)=(p.sptr-=1)
right(p::Parser, l::DepRel)=(arc!(p, p.stack[p.sptr], p.wptr, l); shift(p))
left(p::Parser, l::DepRel)=(arc!(p, p.wptr, p.stack[p.sptr], l); reduce(p))

# Specify the number of gold arcs that become unreachable for each move

function shiftcost(p::Parser, head::AbstractArray, n0l::Integer, s0r::Integer)
    # n0 gets no more ldeps or lhead
    (n0l + (findprev(p.stack, head[p.wptr], p.sptr) > 0))
end

function reducecost(p::Parser, head::AbstractArray, n0l::Integer, s0r::Integer)
    # s0 gets no more rdeps
    return s0r
end

function leftcost(p::Parser, head::AbstractArray, n0l::Integer, s0r::Integer)
    # s0 gets no more rdeps, rhead>n0, 0head
    s0 = p.stack[p.sptr]; s0h = head[s0]
    (s0r + (s0h > p.wptr) + (s0h == 0))
end

function rightcost(p::Parser, head::AbstractArray, n0l::Integer, s0r::Integer)
    # n0 gets no more ldeps, rhead, 0head, or lhead<s0
    n0 = p.wptr; n0h = head[n0]
    (n0l + (n0h > n0) + (n0h == 0) + (findprev(p.stack, n0h, p.sptr-1) > 0))
end

# The cost of a move is the number of gold arcs that become unreachable
# as a result of the move (Goldberg and Nivre 2013).  We provide two
# functions, eagercosts() and hybridcosts() to use as templates for
# movecosts().

function eagercosts(p::Parser, head::AbstractArray, deprel::AbstractArray, cost::Pvec)
    @assert (length(head) == p.nword)
    @assert (length(deprel) == p.nword)
    @assert (length(cost) == p.nmove)
    fill!(cost, Pinf)
    n0 = p.wptr
    s0 = (p.sptr >= 1 ? p.stack[p.sptr] : 0)
    n0l=0; for i=1:p.sptr; si=p.stack[i]; head[si]==n0 && p.head[si]==0 && (n0l+=1); end
    s0r=0; for i=p.wptr:p.nword; (head[i]==s0) && (s0r += 1); end

    shiftok(p)  && (cost[shiftmove(p)]  =  shiftcost(p, head, n0l, s0r))
    reduceok(p) && (cost[reducemove(p)] = reducecost(p, head, n0l, s0r))
    if leftok(p)                                                # left adds the arc (n0,s0)
        lcost = leftcost(p, head, n0l, s0r)
        if (head[s0] == n0)                                     # if this is the correct head
            cost[leftmoves(p)] = lcost + 1                      # +1 for the wrong labels
            cost[leftmove(p,deprel[s0])] -= 1                   # except for the correct label
        else                                                    
            cost[leftmoves(p)] = lcost				# otherwise we incur leftcost
        end
    end
    if rightok(p)                                               # right adds the arc (s0,n0)
        rcost = rightcost(p, head, n0l, s0r)
        if (head[n0] == s0)                                     # if this is the correct head
            cost[rightmoves(p)] = rcost+1  			# +1 for the wrong labels
            cost[rightmove(p,deprel[n0])] -= 1                  # except for the correct label
        else                                                    # 
            cost[rightmoves(p)] = rcost 			# otherwise we incur rightcost
        end
    end
    return cost
end # eagercosts


function hybridcosts(p::Parser, head::AbstractArray, deprel::AbstractArray, cost::Pvec)
    @assert (length(head) == p.nword)
    @assert (length(deprel) == p.nword)
    @assert (length(cost) == p.nmove)
    fill!(cost, Pinf)
    n0 = p.wptr
    s0 = (p.sptr > 0 ? p.stack[p.sptr] : 0)
    s1 = (p.sptr > 1 ? p.stack[p.sptr-1] : 0)
    n0l=0; for i=1:p.sptr; si=p.stack[i]; head[si]==n0 && p.head[si]==0 && (n0l+=1); end
    s0r=0; for i=p.wptr:p.nword; (head[i]==s0) && (s0r += 1); end

    shiftok(p)  && (cost[shiftmove(p)]  =  shiftcost(p, head, n0l, s0r))
    reduceok(p) && (cost[reducemove(p)] = reducecost(p, head, n0l, s0r))
    if leftok(p)                                                # left adds the arc (n0,s0)
        lcost = leftcost(p, head, n0l, s0r)
        if (head[s0] == n0)                                     # if this is the correct head
            cost[leftmoves(p)] = lcost + 1                      # +1 for the wrong labels
            cost[leftmove(p,deprel[s0])] -= 1                   # except for the correct label
        else                                                    
            cost[leftmoves(p)] = lcost				# otherwise we incur leftcost
        end
    end
    if rightok(p)                                               # right adds the arc (s1,s0)
        rcost = rightcost(p, head, n0l, s0r)
        if (head[s0] == s1)                                     # if this is the correct head
            cost[rightmoves(p)] = rcost+1  			# +1 for the wrong labels
            cost[rightmove(p,deprel[s0])] -= 1                  # except for the correct label
        else                                                    # 
            cost[rightmoves(p)] = rcost 			# otherwise we incur rightcost
        end
    end
    return cost
end # hybridcosts



function arc!(p::Parser, h::Position, d::Position, l::DepRel)
    p.head[d] = h
    p.deprel[d] = l
    if d < h
        p.lcnt[h] += 1
        p.ldep[h, p.lcnt[h]] = d
    else
        p.rcnt[h] += 1
        p.rdep[h, p.rcnt[h]] = d
    end
end

import Base: copy!, isequal

function copy!(dst::Parser, src::Parser)
    @assert dst.nword == src.nword
    @assert dst.ndeps == src.ndeps
    @assert dst.nmove == src.nmove
    dst.wptr = src.wptr
    dst.sptr = src.sptr
    copy!(dst.stack, src.stack)
    copy!(dst.head, src.head)
    copy!(dst.deprel, src.deprel)
    copy!(dst.lcnt, src.lcnt)
    copy!(dst.rcnt, src.rcnt)
    copy!(dst.ldep, src.ldep)
    copy!(dst.rdep, src.rdep)
    dst
end # copy!

function isequal(a::Parser, b::Parser)
    f = fieldnames(a)
    all(map(isequal, map(n->a.(n), f), map(n->b.(n), f)))
end

