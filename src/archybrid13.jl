# archybrid13.jl, Deniz Yuret, March 28, 2015
# Transition based greedy arc-hybrid parser based on:
# [GN13]  Goldberg, Yoav; Nivre, Joakim. Training Deterministic Parsers with Non-Deterministic Oracles. TACL 2013.
# [H13]   http://honnibal.wordpress.com/2013/12/18/a-simple-fast-algorithm-for-natural-language-dependency-parsing
# [KGS11] Kuhlmann, Marco, Carlos Gómez-Rodríguez, and Giorgio Satta. Dynamic programming algorithms for transition-based dependency parsers. ACL 2011.

typealias ArcHybrid13 Parser{:ArcHybrid13}

# In the arc-hybrid system (KGS11), a configuration c= (σ,β,A) consists of
# a stack σ, a buffer β, and a set A of dependency arcs.

# There are three types of moves:
# SHIFT[(σ, b|β, A)] = (σ|b, β, A)
# RIGHT_lb[(σ|s1|s0, β, A)] = (σ|s1, β, A ∪ {(s1, lb, s0)})
# LEFT_lb[(σ|s, b|β, A)] = (σ, b|β, A ∪ {(b, lb, s)})

# In (GN13) the initial configuration has an empty stack, and a buffer
# with special symbol ROOT to the left of all the words.  We don't use
# an explicit ROOT, instead we use REDUCE when there is a single
# word in the stack to get the same effect.

# We comment out the definitions identical to the ones in parser.jl.

# shift(p::Parser)=(p.stack[p.sptr+=1]=p.wptr; p.wptr+=1)
# reduce(p::Parser)=(p.sptr-=1)
# left(p::Parser, l::DepRel)=(arc!(p, p.wptr, p.stack[p.sptr], l); reduce(p))
right(p::ArcHybrid13, l::DepRel)=(arc!(p, p.stack[p.sptr-1], p.stack[p.sptr], l); reduce(p))

# Moves are represented by integers 1..p.nmove.
# They correspond to REDUCE,L1,R1,L2,R2,...,L[ndeps],R[ndeps],SHIFT

# reducemove(p::Parser)=1
# leftmoves(p::Parser)=(2:2:(p.nmove-2))
# rightmoves(p::Parser)=(3:2:(p.nmove-1))
# shiftmove(p::Parser)=p.nmove

# Dependency labels (deprel) are represented by integers 1..p.ndeps
# The special ROOT deprel is represented by 0.

# leftmove(p::Parser,l::DepRel)=(l<<1)
# rightmove(p::Parser,l::DepRel)=(1+l<<1)
# label(p::Parser,m::Move)=convert(DepRel,m>>1)

# The only legal first move is SHIFT.  The only legal last move is
# REDUCE (linking the last word to the ROOT).  We perform the first
# SHIFT during initialization so our initial state is [w1][w2,...,wn].
# We don't perform the last REDUCE so our final state is [wi][].  This
# ensures 2n-2 moves for each sentence.

# init!(p::Parser)=(p.nmove=(2+p.ndeps<<1);shift(p))

# GN13 has the following preconditions for moves: "There is a
# precondition on RIGHT to be legal only when the stack has at least two
# elements, and on LEFT to be legal only when the stack is non-empty and
# s != ROOT."  We introduce the additional REDUCE move
# when the stack has a single word to represent the ROOT linkage.  We
# terminate with the last word in stack, so REDUCE has a precondition of
# a non-empty buffer.

shiftok(p::ArcHybrid13)=(p.wptr <= p.nword)
rightok(p::ArcHybrid13)=(p.sptr > 1)
leftok(p::ArcHybrid13)=((p.wptr <= p.nword) && (p.sptr > 0))
reduceok(p::ArcHybrid13)=((p.sptr == 1) && (p.wptr <= p.nword))
anyvalidmoves(p::ArcHybrid13)=((p.wptr <= p.nword) || (p.sptr > 1))

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

function movecosts(p::ArcHybrid13, head::AbstractArray, deprel::AbstractArray, cost::Pvec=Array(Position,p.nmove))
    hybridcosts(p,head,deprel,cost)
end

function shiftcost(p::ArcHybrid13, head::AbstractArray, n0l::Integer, s0r::Integer)
    # n0 gets no more ldeps or lhead<s0 or root head if there is s0
    n0 = p.wptr; n0h = head[n0]
    (n0l + (findprev(p.stack, n0h, p.sptr-1) > 0) + ((n0h==0) && (p.sptr>0)))
end

function reducecost(p::ArcHybrid13, head::AbstractArray, n0l::Integer, s0r::Integer)
    # s0 gets no more rdeps or rhead
    s0 = p.stack[p.sptr]; s0h = head[s0]
    (s0r + (s0h >= p.wptr))
end

function leftcost(p::ArcHybrid13, head::AbstractArray, n0l::Integer, s0r::Integer)
    # s0 gets no more rdeps, rhead>n0, s1 head or 0head if alone
    s0 = p.stack[p.sptr]; s0h = head[s0]
    (s0r + ((s0h > p.wptr) || ((p.sptr == 1) && (s0h == 0)) || ((p.sptr > 1) && (s0h == p.stack[p.sptr-1]))))
end

function rightcost(p::ArcHybrid13, head::AbstractArray, n0l::Integer, s0r::Integer)
    # s0 gets no more rdeps or rhead
    s0 = p.stack[p.sptr]; s0h = head[s0]
    (s0r + (s0h >= p.wptr))
end

