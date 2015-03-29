# arceager.jl, Deniz Yuret, March 27, 2015
# Transition based greedy arc-eager parser, modified to have a single ROOT word.
# [GN13] Goldberg, Yoav; Nivre, Joakim. Training Deterministic Parsers with Non-Deterministic Oracles. TACL 2013.
# [N03]  Nivre, Joakim. An efficient algorithm for projective dependency parsing. IWPT 2003.
# [H13]  http://honnibal.wordpress.com/2013/12/18/a-simple-fast-algorithm-for-natural-language-dependency-parsing

typealias ArcEager Parser{:ArcEager}

# In the arc-eager system (N03), a configuration c= (σ,β,A) consists of
# a stack σ, a buffer β, and a set A of dependency arcs.

# There are four types of moves:
# SHIFT[(σ, b|β, A)] = (σ|b, β, A)
# REDUCE[(σ|s, β, A)] = (σ, β, A)
# RIGHT_lb[(σ|s, b|β, A)] = (σ|s|b, β, A∪{(s,lb,b)})
# LEFT_lb[(σ|s, b|β, A)] = (σ, b|β, A∪{(b,lb,s)})

# We comment out the definitions that are identical to ones in parser.jl.
# shift(p::ArcEager)=(p.stack[p.sptr+=1]=p.wptr; p.wptr+=1)
# reduce(p::ArcEager)=(p.sptr-=1)
# right(p::ArcEager, l::DepRel)=(arc!(p, p.stack[p.sptr], p.wptr, l); shift(p))
# left(p::ArcEager, l::DepRel)=(arc!(p, p.wptr, p.stack[p.sptr], l); reduce(p))

# Moves are represented by integers 1..p.nmove
# They correspond to REDUCE,L1,R1,L2,R2,..,L[ndeps],R[ndeps],SHIFT

# reducemove(p::ArcEager)=1
# leftmoves(p::ArcEager)=(2:2:(p.nmove-2))
# rightmoves(p::ArcEager)=(3:2:(p.nmove-1))
# shiftmove(p::ArcEager)=p.nmove

# Dependency labels (deprel) are represented by integers 1..p.ndeps
# The special ROOT deprel is represented by 0.  We use REDUCE for the
# LEFT move from the ROOT token.

# leftmove(p::ArcEager,l::DepRel)=(l<<1)
# rightmove(p::ArcEager,l::DepRel)=(1+l<<1)
# label(p::ArcEager,m::Move)=convert(DepRel,m>>1)

# In GN13 the initial configuration has an empty stack, and a buffer
# with special symbol ROOT to the right of all the words at w[n+1], i.e.
# [][w1,w2,...,wn,ROOT]. The final state in GN13 is only the ROOT token
# in the buffer and nothing on the stack, i.e. [][ROOT].  The special
# ROOT token can have more than one child, thus the sentence does not
# always have a unique rootword.

# In this implementation we will enforce a unique rootword.  We don't
# use an explicit ROOT token.  Our final state is the single root word
# at s0, and an empty buffer i.e. [rootword][].  The unique rootword
# is marked with head=0 and deprel=0.  The only legal first move,
# SHIFT, is performed at initialization so the initial configuration
# is [w1][w2,...,wn].  This ensures 2n-2 moves for each sentence.

# init!(p::ArcEager)=(p.nmove=(2+p.ndeps<<1);shift(p))

# GN13 has the following preconditions for moves: "There is a
# precondition on the RIGHT and SHIFT transitions to be legal only when
# b != ROOT (p.wptr <= p.nword), and for LEFT, RIGHT and REDUCE to be
# legal only when the stack is non-empty (p.sptr >= 1). Moreover, LEFT
# is only legal when s does not have a parent in A, and REDUCE when s
# does have a parent in A."

# GN13 ends up with multiple rootwords by shifting them to the stack and
# popping them with left moves when the only token left in the buffer is
# ROOT.  We can prevent multiple rootwords by:
# (1) avoid SHIFT on last buffer word unless the stack is empty.
# (2) avoid RIGHT on last buffer word if multiple headless words in stack.

shiftok(p::ArcEager)=((p.wptr < p.nword) || ((p.wptr == p.nword) && (p.sptr == 0)))
rightok(p::ArcEager)=((p.sptr > 0) && ((p.wptr < p.nword) || ((p.wptr == p.nword) && (headless(p)==1))))
leftok(p::ArcEager)=((p.sptr > 0) && (p.wptr <= p.nword) && (p.head[p.stack[p.sptr]] == 0))
reduceok(p::ArcEager)=((p.sptr > 0) && (p.head[p.stack[p.sptr]] != 0))
headless(p::ArcEager)=(h=0;for i=1:p.sptr; si=p.stack[i]; (p.head[si]==0) && (h+=1); end; h)

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

anyvalidmoves(p::ArcEager)=((p.wptr <= p.nword) || ((p.sptr > 0) && (p.head[p.stack[p.sptr]] != 0)))

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

# function shiftcost(p::ArcEager, head::AbstractArray, deprel::AbstractArray)
#     # n0 gets no more ldeps or lhead
#     n0 = p.wptr; n0h = head[n0]
#     n0l = 0; (for i=1:p.sptr; si=p.stack[i]; (head[si]==n0) && (p.head[si]==0) && (n0l+=1); end)
#     (n0l + (findprev(p.stack, n0h, p.sptr) > 0)) 
# end

# function reducecost(p::ArcEager, head::AbstractArray, deprel::AbstractArray)
#     # s0 gets no more rdeps
#     s0 = p.stack[p.sptr]; s0r = 0
#     for i=p.wptr:p.nword; (head[i]==s0) && (s0r += 1); end
#     return s0r
# end

# function leftcost(p::ArcEager, head::AbstractArray, deprel::AbstractArray)
#     # s0 gets no more rdeps, rhead>n0, 0head
#     s0 = p.stack[p.sptr]; s0h = head[s0]; s0r = 0
#     for i=p.wptr:p.nword; (head[i]==s0) && (s0r += 1); end
#     (s0r + (s0h > p.wptr) + (s0h == 0))
# end

# function rightcost(p::ArcEager, head::AbstractArray, deprel::AbstractArray)
#     # n0 gets no more ldeps, rhead, 0head, or lhead<s0
#     n0 = p.wptr; n0h = head[n0]; n0l = 0
#     for i=1:p.sptr; si=p.stack[i]; (head[si]==n0) && (p.head[si]==0) && (n0l+=1); end
#     (n0l + (n0h > n0) + (n0h == 0) + (findprev(p.stack, n0h, p.sptr-1) > 0))
# end

