# parser.jl, Deniz Yuret, March 30, 2015
#
# Transition based parser based on:
#
# [GN13]  Goldberg, Yoav; Nivre, Joakim. Training Deterministic Parsers with Non-Deterministic Oracles. TACL 2013.
# [H13]   http://honnibal.wordpress.com/2013/12/18/a-simple-fast-algorithm-for-natural-language-dependency-parsing
# [KGS11] Kuhlmann, Marco, Carlos Gómez-Rodríguez, and Giorgio Satta. Dynamic programming algorithms for transition-based dependency parsers. ACL 2011
# [N03]   Nivre, Joakim. An efficient algorithm for projective dependency parsing. IWPT 2003.

# We define Parser{T,V} as a parametric type where T is :ArcHybrid,
# :ArcEager etc. and V is a variant symbol such as :GN13, :R1.  This
# allows specialization and inheritence of methods.  Thanks to
# julia-users members Toivo Henningsson and Simon Danisch for suggesting
# this design.


@compat typealias Position UInt8
@compat typealias DepRel UInt8
typealias Move Integer
typealias Pvec AbstractVector{Position}
typealias Dvec AbstractVector{DepRel}
typealias Pmat AbstractMatrix{Position}
const Pinf=typemax(Position)
Pzeros(n::Integer...)=zeros(Position, n...)
Dzeros(n::Integer...)=zeros(DepRel, n...)


type Parser{T,V}
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

# A parser provides four functions: 
#
# nmoves(p,[s]): how many moves it has or takes to parse a sentence or a corpus
# anyvalidmoves(p): whether there are any valid moves in current state
# move!(p,m): execute a particular move
# movecosts(p,h,d,c): the number of gold arcs that become unreachable for each move
#
# Default definitions are provided below.  A new parser Parser{T} will
# fall back to these if it doesn't override them.

# By default our initial state is [w1][w2,...,wn] and our final state is
# [wroot][].  This is achieved by always performing a SHIFT during
# initialization and ensures 2n-2 moves for each sentence.

init!(p::Parser)=(p.nmove=(2+p.ndeps<<1);shift(p))
nmoves(p::Parser)=p.nmove
nmoves(p::Parser,s::Sentence)=((wcnt(s)-1)<<1)
nmoves(p::Parser,c::Corpus)=(n=0; for s in c; n += nmoves(p,s); end; n)

anyvalidmoves(p::Parser)=(shiftok(p)||reduceok(p)||leftok(p)||rightok(p))

function move!(p::Parser, m::Move)
    @assert (1 <= m <= p.nmove) "Move $m is not supported"
    (m == shiftmove(p))  ? (@assert shiftok(p);  shift(p)) :
    in(m, rightmoves(p)) ? (@assert rightok(p);  right(p,label(p,m))) :
    in(m, leftmoves(p))  ? (@assert leftok(p);   left(p,label(p,m))) :
    (m == reducemove(p)) ? (@assert reduceok(p); reduce(p)) :
    error("Move $m is not supported")
end

function movecosts(p::Parser, head::AbstractArray, deprel::AbstractArray, 
                   cost::Pvec=Array(Position,p.nmove))
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
    if leftok(p)                 
        (lh,ld) = leftmovepair(p)                               # left adds the arc (lh,ld)
        lcost = leftcost(p, head, n0l, s0r)
        if (head[ld] == lh)                                     # if this is the correct head
            cost[leftmoves(p)] = lcost + 1                      # +1 for the wrong labels
            cost[leftmove(p,deprel[ld])] -= 1                   # except for the correct label
        else                                                    
            cost[leftmoves(p)] = lcost				# otherwise we incur leftcost
        end
    end
    if rightok(p)
        (rh,rd) = rightmovepair(p)                              # right adds the arc (rh,rd)
        rcost = rightcost(p, head, n0l, s0r)
        if (head[rd] == rh)                                     # if this is the correct head
            cost[rightmoves(p)] = rcost+1  			# +1 for the wrong labels
            cost[rightmove(p,deprel[rd])] -= 1                  # except for the correct label
        else                                                    # 
            cost[rightmoves(p)] = rcost 			# otherwise we incur rightcost
        end
    end
    return cost
end # movecosts


################################ ArcEager,GN13 ##################
# We provide default fallback definitions for Parser based on the
# ArcEager system from [N03,GN13].

typealias ArcEager Parser{:ArcEager}
typealias ArcEager13 Parser{:ArcEager,:GN13}

# In the arc-eager system (N03), a configuration c= (σ,β,A) consists of
# a stack σ, a buffer β, and a set A of dependency arcs.

# There are four types of moves:
# SHIFT[(σ, b|β, A)] = (σ|b, β, A)
# REDUCE[(σ|s, β, A)] = (σ, β, A)
# RIGHT_lb[(σ|s, b|β, A)] = (σ|s|b, β, A∪{(s,lb,b)})
# LEFT_lb[(σ|s, b|β, A)] = (σ, b|β, A∪{(b,lb,s)})

leftmovepair(p::Parser)=(p.wptr,p.stack[p.sptr])
rightmovepair(p::Parser)=(p.stack[p.sptr],p.wptr)

shift(p::Parser)=(p.stack[p.sptr+=1]=p.wptr; p.wptr+=1)
reduce(p::Parser)=(p.sptr-=1)
left(p::Parser, l::DepRel)=(arc!(p, p.wptr, p.stack[p.sptr], l); reduce(p))
right(p::Parser, l::DepRel)=(arc!(p, p.stack[p.sptr], p.wptr, l); shift(p))

# Moves are represented by integers 1..p.nmove
# Default order is REDUCE,L1,R1,L2,R2,..,L[ndeps],R[ndeps],SHIFT

shiftmove(p::Parser)=p.nmove
reducemove(p::Parser)=1
rightmoves(p::Parser)=(3:2:(p.nmove-1))
leftmoves(p::Parser)=(2:2:(p.nmove-2))

# Dependency labels (deprel) are represented by integers 1..p.ndeps.
# The special ROOT deprel is represented by 0.

rightmove(p::Parser,l::DepRel)=(1+l<<1)
leftmove(p::Parser,l::DepRel)=(l<<1)
label(p::Parser,m::Move)=convert(DepRel,m>>1)

# GN13 has the following preconditions for moves:
#
# "There is a precondition on the RIGHT and SHIFT transitions to be
# legal only when b != ROOT (p.wptr <= p.nword), and for LEFT, RIGHT and
# REDUCE to be legal only when the stack is non-empty (p.sptr >
# 0). Moreover, LEFT is only legal when s does not have a parent in A,
# and REDUCE when s does have a parent in A."
# 
# We implement the ROOT-LEFT as a REDUCE, our LEFT gets an extra
# condition (p.wptr <= p.nword) and REDUCE gets an extra option.

shiftok(p::Parser)=(p.wptr <= p.nword)
reduceok(p::Parser)=((p.sptr > 0) && (s0head(p) || s0head0(p)))
rightok(p::Parser)=((p.wptr <= p.nword) && (p.sptr > 0))
leftok(p::Parser)=((p.wptr <= p.nword) && (p.sptr > 0) && !s0head(p))

s0head(p::Parser)=(p.head[p.stack[p.sptr]] != 0)
s0head0(p::Parser)=((p.wptr > p.nword) && (p.sptr > 1))

# movecosts() counts gold arcs that become impossible after each move.
# A token starts life without any arcs in the buffer.  It moves to the
# head of the buffer (n0) with shift or right moves (each right ends
# with a shift).  Left deps are acquired first while at n0 using left
# moves (each left move ends with a reduce) possibly interspersed with
# other reduces (to get rid of s0 that already have heads).  The token
# moves to s0 with a right or shift.  Then rdeps are acquired while at
# s0 using right moves.  Head is acquired as n0 before rdeps (which
# moves the token to s0, so buffer words never have heads) or as s0
# after rdeps.

function shiftcost(p::Parser, head::AbstractArray, n0l::Integer, s0r::Integer)
    # eager: n0 gets no more ldeps or lhead
    (n0l + (0 < findprev(p.stack, head[p.wptr], p.sptr)))
end

function reducecost(p::Parser, head::AbstractArray, n0l::Integer, s0r::Integer)
    s0r # eager: s0 gets no more rdeps
end

function leftcost(p::Parser, head::AbstractArray, n0l::Integer, s0r::Integer)
    # eager: s0 gets no more rdeps, rhead>n0, 0head
    s0 = p.stack[p.sptr]; s0h = head[s0]
    (s0r + (s0h > p.wptr) + (s0h == 0))
end

function rightcost(p::Parser, head::AbstractArray, n0l::Integer, s0r::Integer)
    # eager: n0 gets no more ldeps, rhead, 0head, or lhead<s0
    n0 = p.wptr; n0h = head[n0]
    (n0l + (n0h > n0) + (n0h == 0) + (0 < findprev(p.stack, n0h, p.sptr-1)))
end


################################ ArcEagerR1 ##########################
# ArcEagerR1 is a modification of ArcEager to ensure a single rootword

typealias ArcEagerR1 Parser{:ArcEager,:R1}

# In GN13 the initial configuration has an empty stack, and a buffer
# with special symbol ROOT to the right of all the words at w[n+1], i.e.
# [][w1,w2,...,wn,ROOT]. The final state in GN13 is only the ROOT token
# in the buffer and nothing on the stack, i.e. [][ROOT].  The special
# ROOT token can have more than one child, thus the sentence does not
# always have a unique rootword.

# GN13 ends up with multiple rootwords by shifting them to the stack and
# popping them with left moves when the only token left in the buffer is
# ROOT.  We can prevent multiple rootwords by:
# (1) avoid SHIFT on last buffer word unless the stack is empty.
# (2) avoid RIGHT on last buffer word if multiple headless words in stack.

shiftok(p::ArcEagerR1)=((p.wptr < p.nword) || ((p.wptr == p.nword) && (p.sptr == 0)))
rightok(p::ArcEagerR1)=((p.sptr > 0) && ((p.wptr < p.nword) || ((p.wptr == p.nword) && (headless(p)==1))))
leftok(p::ArcEagerR1)=((p.sptr > 0) && (p.wptr <= p.nword) && (p.head[p.stack[p.sptr]] == 0))
reduceok(p::ArcEagerR1)=((p.sptr > 0) && (p.head[p.stack[p.sptr]] != 0))
headless(p::ArcEagerR1)=(h=0;for i=1:p.sptr; si=p.stack[i]; (p.head[si]==0) && (h+=1); end; h)

# Proof: Buffer words never have heads, some stack words do.  As long as
# we have words in the buffer we can recover a single root.  Once the
# buffer is empty, nothing can go back into it.  With an empty buffer,
# REDUCE is the only possible move, (we don't allow left moves with an
# empty buffer) now new non-root arcs can be added.  So before we get to
# the empty buffer we need to make sure the stack will contain a single
# headless rootword.  Stack words are either headless or are next to
# their parents, i.e. a non-empty stack always contains at least one
# headless word.  SHIFTing the last word into a non-empty stack would
# add another, so can't do it.  RIGHT with the last word would not
# create another headless stack word, but we have to make sure there is
# only one before doing RIGHT on the last word.

# WolframAlpha suggested the following succinct form for anyvalidmoves(p).
# REDUCE if ((p.sptr > 0) && (p.head[p.stack[p.sptr]] != 0))
# otherwise (p.wptr <= p.nword) is true
# SHIFT if (p.sptr == 0)
# otherwise (p.sptr > 0) is true
# so (p.head[p.stack[p.sptr]] != 0) is false
# and LEFT is possible

anyvalidmoves(p::ArcEagerR1)=((p.wptr <= p.nword) || ((p.sptr > 0) && (p.head[p.stack[p.sptr]] != 0)))


################################ ArcHybrid ####################
# ArcHybrid13 is the ArcHybrid system described in [KGS11,GN13]

typealias ArcHybrid Parser{:ArcHybrid}
typealias ArcHybrid13 Parser{:ArcHybrid,:GN13}

# In the arc-hybrid system (KGS11), a configuration c= (σ,β,A) consists
# of a stack σ, a buffer β, and a set A of dependency arcs.

# There are three types of moves:
# SHIFT[(σ, b|β, A)] = (σ|b, β, A)
# RIGHT_lb[(σ|s1|s0, β, A)] = (σ|s1, β, A ∪ {(s1, lb, s0)})
# LEFT_lb[(σ|s, b|β, A)] = (σ, b|β, A ∪ {(b, lb, s)})

# The hybrid system has a different RIGHT action linking (s1,s0).

right(p::ArcHybrid,l::DepRel)=(arc!(p, p.stack[p.sptr-1], p.stack[p.sptr], l); reduce(p))
rightmovepair(p::ArcHybrid)=(p.stack[p.sptr-1], p.stack[p.sptr])

# GN13 has the following preconditions for ArcHybrid moves: 
#
# "There is a precondition on RIGHT to be legal only when the stack has
# at least two elements, and on LEFT to be legal only when the stack is
# non-empty and s != ROOT."

shiftok(p::ArcHybrid)=(p.wptr <= p.nword)
rightok(p::ArcHybrid)=(p.sptr > 1)
leftok(p::ArcHybrid)=((p.wptr <= p.nword) && (p.sptr > 0))

# We introduce an additional REDUCE move when the stack has a single
# word to represent the ROOT linkage.  We terminate with the last word
# in stack, so REDUCE has a precondition of a non-empty buffer.

reduceok(p::ArcHybrid)=((p.sptr == 1) && (p.wptr <= p.nword))
anyvalidmoves(p::ArcHybrid)=((p.wptr <= p.nword) || (p.sptr > 1))

# movecosts() counts gold arcs that become impossible after possible
# moves.  Tokens start their lifecycle in the buffer without links.
# They move to the top of the buffer (n0) with SHIFT moves.  There they
# acquire left dependents using LEFT moves.  After that a single SHIFT
# moves them to the top of the stack (s0).  There they acquire right
# dependents with SHIFT-RIGHT pairs.  Finally from s0 they acquire a
# head with a LEFT or RIGHT move.  Any token from the buffer may become
# the right head but only s1 from the stack may become a left head for
# s0.  The parser terminates with a single word at s0 whose head is ROOT
# (represented as head=deprel=0).
#
# 1. SHIFT moves n0 to s0: n0 cannot acquire left dependents after a
# shift.  Also it can no longer get a head from the stack to the left
# of s0 or get a root head if there is s0: (0+s\s0,n0) + (n0,s)
#
# 2. R0MOVE pops s0 linking it to ROOT: s0 cannot acquire a head or
# dependent from the buffer after right: (s0,b) + (b,s0)
#
# 3. RIGHT adds (s1,s0): s0 cannot acquire a head or dependent from
# the buffer after right: (s0,b) + (b,s0)
#
# 4. LEFT adds (n0,s0): s0 cannot acquire s1 or 0 (if there is no s1)
# or ni (i>0) as head.  It also cannot acquire any more right
# children: (s0,b) + (b\n0,s0) + (s1 or 0,s0)

function shiftcost(p::ArcHybrid, head::AbstractArray, n0l::Integer, s0r::Integer)
    # n0 gets no more ldeps or lhead<s0 or root head if there is s0
    n0 = p.wptr; n0h = head[n0]
    (n0l + (findprev(p.stack, n0h, p.sptr-1) > 0) + ((n0h==0) && (p.sptr>0)))
end

function reducecost(p::ArcHybrid, head::AbstractArray, n0l::Integer, s0r::Integer)
    # s0 gets no more rdeps or rhead
    s0 = p.stack[p.sptr]; s0h = head[s0]
    (s0r + (s0h >= p.wptr))
end

function leftcost(p::ArcHybrid, head::AbstractArray, n0l::Integer, s0r::Integer)
    # s0 gets no more rdeps, rhead>n0, s1 head or 0head if alone
    s0 = p.stack[p.sptr]; s0h = head[s0]
    (s0r + ((s0h > p.wptr) || ((p.sptr == 1) && (s0h == 0)) || ((p.sptr > 1) && (s0h == p.stack[p.sptr-1]))))
end

function rightcost(p::ArcHybrid, head::AbstractArray, n0l::Integer, s0r::Integer)
    # s0 gets no more rdeps or rhead
    s0 = p.stack[p.sptr]; s0h = head[s0]
    (s0r + (s0h >= p.wptr))
end

################################ ArcHybridR1 #############################
# ArcHybridR1 is a modification of ArcHybrid to ensure a single rootword

typealias ArcHybridR1 Parser{:ArcHybrid,:R1}

# It turns out we can ensure a single rootword by simply not allowing
# the reduce move (i.e. the ROOT-RIGHT move)

reduceok(p::ArcHybridR1)=false
reducemove(p::ArcHybridR1)=nothing

# This gives us one less legal move.

init!(p::ArcHybridR1)=(p.nmove=(1+p.ndeps<<1);shift(p))

# Moves are represented by integers 1..p.nmove, 0 is not valid.
# They correspond to L1,R1,L2,R2,...,L[ndeps],R[ndeps],SHIFT

shiftmove(p::ArcHybridR1)=p.nmove
leftmoves(p::ArcHybridR1)=(1:2:(p.nmove-2))
rightmoves(p::ArcHybridR1)=(2:2:(p.nmove-1))

# Dependency labels (deprel) are represented by integers 1..p.ndeps
# The special ROOT deprel is represented by 0.

leftmove(p::ArcHybridR1, l::DepRel)=(l<<1-1)
rightmove(p::ArcHybridR1, l::DepRel)=(l<<1)
label(p::ArcHybridR1, m::Move)=convert(DepRel,(m+1)>>1)



################################################################
# Some general utility functions:

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

function Base.isequal(a::Parser, b::Parser)
    f = fieldnames(a)
    all(map(isequal, map(n->a.(n), f), map(n->b.(n), f)))
end

function Base.copy!(dst::Parser, src::Parser)
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

function reset!(p::Parser)
    p.wptr = 1
    p.sptr = 0
    p.stack[:] = 0
    p.head[:] = 0
    p.deprel[:] = 0
    p.lcnt[:] = 0
    p.rcnt[:] = 0
    p.ldep[:] = 0
    p.rdep[:] = 0
    init!(p)
end

reset!{T<:Parser}(pa::Vector{T})=(for p in pa; reset!(p); end)

Base.show(io::IO, p::Parser)=print(io, map(int,p.stack[1:p.sptr]), Int[p.wptr], map(int,p.head))
