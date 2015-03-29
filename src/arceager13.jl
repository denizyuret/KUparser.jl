# arceager13.jl, Deniz Yuret, March 28, 2015
# Transition based greedy arc-eager parser based on:
# [GN13] Goldberg, Yoav; Nivre, Joakim. Training Deterministic Parsers with Non-Deterministic Oracles. TACL 2013.
# [N03]  Nivre, Joakim. An efficient algorithm for projective dependency parsing. IWPT 2003.
# [H13]  http://honnibal.wordpress.com/2013/12/18/a-simple-fast-algorithm-for-natural-language-dependency-parsing

typealias ArcEager13 Parser{:ArcEager13}

# In the arc-eager system (N03), a configuration c= (σ,β,A) consists of
# a stack σ, a buffer β, and a set A of dependency arcs.

# There are four types of moves:
# SHIFT[(σ, b|β, A)] = (σ|b, β, A)
# REDUCE[(σ|s, β, A)] = (σ, β, A)
# RIGHT_lb[(σ|s, b|β, A)] = (σ|s|b, β, A∪{(s,lb,b)})
# LEFT_lb[(σ|s, b|β, A)] = (σ, b|β, A∪{(b,lb,s)})

# We comment out the definitions that are identical to ones in parser.jl.
# shift(p::Parser)=(p.stack[p.sptr+=1]=p.wptr; p.wptr+=1)
# reduce(p::Parser)=(p.sptr-=1)
# right(p::Parser, l::DepRel)=(arc!(p, p.stack[p.sptr], p.wptr, l); shift(p))
# left(p::Parser, l::DepRel)=(arc!(p, p.wptr, p.stack[p.sptr], l); reduce(p))

# Moves are represented by integers 1..p.nmove
# They correspond to REDUCE,L1,R1,L2,R2,..,L[ndeps],R[ndeps],SHIFT

# reducemove(p::Parser)=1
# leftmoves(p::Parser)=(2:2:(p.nmove-2))
# rightmoves(p::Parser)=(3:2:(p.nmove-1))
# shiftmove(p::Parser)=p.nmove

# Dependency labels (deprel) are represented by integers 1..p.ndeps
# The special ROOT deprel is represented by 0.  We use REDUCE for the
# LEFT move from the ROOT token.

# leftmove(p::Parser,l::DepRel)=(l<<1)
# rightmove(p::Parser,l::DepRel)=(1+l<<1)
# label(p::Parser,m::Move)=convert(DepRel,m>>1)

# In GN13 the initial configuration has an empty stack, and a buffer
# with special symbol ROOT to the right of all the words at w[n+1], i.e.
# [][w1,w2,...,wn,ROOT]. The final state in GN13 is only the ROOT token
# in the buffer and nothing on the stack, i.e. [][ROOT].  The special
# ROOT token can have more than one child, thus the sentence does not
# always have a unique rootword.

# The only legal first move is SHIFT.  The only legal last move is
# REDUCE (representing ROOT LEFT).  We perform the first SHIFT during
# initialization so our initial state is [w1][w2,...,wn,ROOT].  We don't
# perform the last REDUCE so our final state is [wi][ROOT].  This
# ensures 2n-2 moves for each sentence.

# init!(p::Parser)=(p.nmove=(2+p.ndeps<<1);shift(p))

# GN13 has the following preconditions for moves: "There is a
# precondition on the RIGHT and SHIFT transitions to be legal only when
# b != ROOT (p.wptr <= p.nword), and for LEFT, RIGHT and REDUCE to be
# legal only when the stack is non-empty (p.sptr > 0). Moreover, LEFT
# is only legal when s does not have a parent in A, and REDUCE when s
# does have a parent in A."  Since we implement the ROOT-LEFT as a
# REDUCE, our LEFT gets an extra contition (p.wptr <= p.nword) and
# REDUCE gets an extra option.

# shiftok(p::Parser)=(p.wptr <= p.nword)
# rightok(p::Parser)=((p.wptr <= p.nword) && (p.sptr > 0))
# leftok(p::Parser)=((p.wptr <= p.nword) && (p.sptr > 0) && !s0head(p))
# reduceok(p::Parser)=((p.sptr > 0) && (s0head(p) || s0head0(p)))

# s0head(p::Parser)=(p.head[p.stack[p.sptr]] != 0)
# s0head0(p::Parser)=((p.wptr > p.nword) && (p.sptr > 1))

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

# function shiftcost(p::Parser, head::AbstractArray, deprel::AbstractArray)
#     # n0 gets no more ldeps or lhead
#     n0 = p.wptr; n0h = head[n0]
#     n0l = 0; (for i=1:p.sptr; si=p.stack[i]; (head[si]==n0) && (p.head[si]==0) && (n0l+=1); end)
#     (n0l + (findprev(p.stack, n0h, p.sptr) > 0)) 
# end

# function reducecost(p::Parser, head::AbstractArray, deprel::AbstractArray)
#     # s0 gets no more rdeps
#     s0 = p.stack[p.sptr]; s0r = 0
#     for i=p.wptr:p.nword; (head[i]==s0) && (s0r += 1); end
#     return s0r
# end

# function leftcost(p::Parser, head::AbstractArray, deprel::AbstractArray)
#     # s0 gets no more rdeps, rhead>n0, 0head
#     s0 = p.stack[p.sptr]; s0h = head[s0]; s0r = 0
#     for i=p.wptr:p.nword; (head[i]==s0) && (s0r += 1); end
#     (s0r + (s0h > p.wptr) + (s0h == 0))
# end

# function rightcost(p::Parser, head::AbstractArray, deprel::AbstractArray)
#     # n0 gets no more ldeps, rhead, 0head, or lhead<s0
#     n0 = p.wptr; n0h = head[n0]; n0l = 0
#     for i=1:p.sptr; si=p.stack[i]; (head[si]==n0) && (p.head[si]==0) && (n0l+=1); end
#     (n0l + (n0h > n0) + (n0h == 0) + (findprev(p.stack, n0h, p.sptr-1) > 0))
# end
