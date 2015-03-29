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

# Moves are represented by integers 1..p.nmove.
# They correspond to R0,L1,R1,L2,R2,...,L[ndeps],R[ndeps],SHIFT

@inline SHIFT(p::ArcHybrid13)=p.nmove
@inline R0MOVE(p::ArcHybrid13)=1
@inline LMOVES(p::ArcHybrid13)=(2:2:(p.nmove-2))
@inline RMOVES(p::ArcHybrid13)=(3:2:(p.nmove-1))

# Dependency labels (deprel) are represented by integers 1..p.ndeps
# The special ROOT deprel is represented by 0.

@inline LMOVE(p::ArcHybrid13, lab::DepRel)=(lab<<1)
@inline RMOVE(p::ArcHybrid13, lab::DepRel)=(1+lab<<1)
@inline LABEL(p::ArcHybrid13, m::Move)=convert(DepRel,m>>1)

# In (GN13) the initial configuration has an empty stack, and a buffer
# with special symbol ROOT to the left of all the words.  We don't use
# an explicit ROOT, instead we use the R0MOVE when there is a single
# word in the stack to get the same effect.

# The only legal first move is SHIFT.  The only legal last move is
# R0MOVE (linking the last word to the ROOT).  We perform the first
# SHIFT during initialization so our initial state is [w1][w2,...,wn].
# We don't perform the last R0MOVE so our final state is [wi][].  This
# ensures 2n-2 moves for each sentence.

init!(p::ArcHybrid13)=(p.nmove=(2+p.ndeps<<1);move!(p,SHIFT(p)))

# GN13 has the following preconditions for moves: "There is a
# precondition on RIGHT to be legal only when the stack has at least two
# elements, and on LEFT to be legal only when the stack is non-empty and
# s != ROOT."  We introduce the special R0MOVE (similar to a REDUCE)
# when the stack has a single word to represent the ROOT linkage.  We
# terminate with the last word in stack, so R0MOVE has a precondition of
# a non-empty buffer.

@inline SHIFTOK(p::ArcHybrid13)=(p.wptr <= p.nword)
@inline RIGHTOK(p::ArcHybrid13)=(p.sptr > 1)
@inline LEFTOK(p::ArcHybrid13)=((p.sptr > 0) && (p.wptr <= p.nword))
@inline R0MOVEOK(p::ArcHybrid13)=((p.sptr == 1) && (p.wptr <= p.nword))

@inline anyvalidmoves(p::ArcHybrid13)=((p.wptr <= p.nword) || (p.sptr > 1))


function move!(p::ArcHybrid13, m::Move)
    @assert (1 <= m <= p.nmove) "Move $m is not supported"
    if m == SHIFT(p)
        @assert SHIFTOK(p)
        p.sptr += 1
        p.stack[p.sptr] = p.wptr
        p.wptr += 1
    elseif m == R0MOVE(p)
        @assert R0MOVEOK(p)
        p.sptr -= 1
    elseif in(m, RMOVES(p))
        @assert RIGHTOK(p)
        arc!(p, p.stack[p.sptr-1], p.stack[p.sptr], LABEL(p,m))
        p.sptr -= 1
    else # in(m, LMOVES(p))
        @assert LEFTOK(p)
        arc!(p, p.wptr, p.stack[p.sptr], LABEL(p,m))
        p.sptr -= 1
    end
end # move!


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

function movecosts(p::ArcHybrid13, head::AbstractArray, deprel::AbstractArray, 
                   cost::Pvec=Array(Position,p.nmove))
    @assert (length(head) == p.nword)
    @assert (length(deprel) == p.nword)
    @assert (length(cost) == p.nmove)
    fill!(cost, Pinf)
    n0 = p.wptr                                                 # n0 is the next word in buffer
    s0 = (p.sptr > 0 ? p.stack[p.sptr] : 0)                     # s0 is top of stack
    s1 = (p.sptr > 1 ? p.stack[p.sptr-1] : 0)                   # s1 is the stack element before s0
    s0h = (s0 != 0 ? head[s0] : 0)                              # s0h is the actual head of s0
    n0h = (n0 <= p.nword ? head[n0] : 0)                        # n0h is the actual head of n0
    s0r = 0; s0 != 0  && (for i=n0:p.nword; head[i]==s0 && (s0r += 1); end) # s0r is the number of right children for s0

    if SHIFTOK(p)                                               # SHIFT valid if n0, moving n0 to s0
        n0l = 0; (for i=1:p.sptr; si=p.stack[i]; head[si]==n0 && (n0l+=1); end)
        cost[SHIFT(p)] = (n0l +                                 # no more left dependents for n0
                          (findprev(p.stack, n0h, p.sptr-1) > 0) + # no heads to the left of s0 for n0
                          ((n0h == 0) && (p.sptr > 0)))         # no root head for n0 if there is s0
    end
    if R0MOVEOK(p)                                              # R0MOVE pops a singleton s0
        cost[R0MOVE(p)] = s0r + (s0h >= n0)                     # no more right head or dependent for s0
    end
    if RIGHTOK(p)                                               # RIGHT valid if s1 making s1 head of s0
        rcost = s0r + (s0h >= n0)                               # no more right head or dependent for s0
        if (s0h == s1)                                          # if we have the correct head
            cost[RMOVES(p)] = rcost + 1                         # +1 for all the wrong labels
            cost[RMOVE(p,deprel[s0])] -= 1                      # except for the correct label
        else                                                    # 
            cost[RMOVES(p)] = rcost                             # if s1 is not the actual head we are done
        end
    end
    if LEFTOK(p)                                                # LEFT valid if n0, making n0 head of s0
        lcost = (s0r +                                          # no more right children for s0
                 ((s0h > n0) ||                                 # no heads to the right of n0 for s0
                  ((p.sptr == 1) && (s0h == 0)) ||              # no root head for s0 if alone
                  ((p.sptr > 1) && (s0h == s1))))               # no more s1 for head of s0
        if (s0h == n0)                                          # if we have the correct head
            cost[LMOVES(p)] = lcost + 1                         # +1 for all the wrong labels
            cost[LMOVE(p,deprel[s0])] -= 1                      # except for the correct label
        else                                                    #
            cost[LMOVES(p)] = lcost                             # if n0 is not the actual head we are done
        end
    end
    return cost
end # movecosts

