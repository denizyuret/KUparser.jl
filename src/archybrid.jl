# archybrid.jl, Deniz Yuret, July 7, 2014: Transition based greedy arc-hybrid parser based on:
# http://honnibal.wordpress.com/2013/12/18/a-simple-fast-algorithm-for-natural-language-dependency-parsing
# Goldberg, Yoav; Nivre, Joakim. Training Deterministic Parsers with Non-Deterministic Oracles. TACL 2013.
# Modified valid moves to output a single root-child.

immutable ArcHybrid <: Parser
    state::ParserState
    nword::Pval
    ndeps::Dval
    nmove::Mval
    function ArcHybrid(nword::Integer, ndeps::Integer)
        @assert (nword <= (typemax(Pval)-1))    "nword > $(typemax(Pval)-1)"
        @assert (ndeps <= (typemax(Mval)-1)>>1) "ndeps > $((typemax(Mval)-1)>>1)"
        a = new(ParserState(nword), nword, ndeps, 1+2*ndeps)
        move!(a, SHIFT)         # only possible first move
        return a
    end
end # ArcHybrid

# move!(p,mv) executes the move mv on parser p.
# In the archybrid system we have three type of moves:
# SHIFT[(σ, b|β, A)] = (σ|b, β, A)
# RIGHT_lb[(σ|s1|s0, β, A)] = (σ|s1, β, A ∪ {(s1, lb, s0)})
# LEFT_lb[(σ|s, b|β, A)] = (σ, b|β, A ∪ {(b, lb, s)})

function move!(p::ArcHybrid, m::Integer)
    @assert (1 <= m <= p.nmove) "Move $m is not supported"
    s = p.state
    if isshift(p,m)
        @assert (s.wptr <= p.nword)
        s.sptr += 1
        s.stack[s.sptr] = s.wptr
        s.wptr += 1
    elseif mdir(m) == RIGHT
        @assert (s.sptr >= 2)
        arc!(s, s.stack[s.sptr-1], s.stack[s.sptr], movedep(m))
        s.sptr -= 1
    else # mdir(m) == LEFT
        @assert ((s.sptr >= 1) && (s.wptr <= p.nword))
        arc!(s, s.wptr, s.stack[s.sptr], movedep(m))
        s.sptr -= 1
    end
end # move!


# movecosts() counts gold arcs that become impossible after possible
# transitions.  Tokens start their lifecycle in the buffer without
# links.  They move to the top of the buffer (n0) with SHIFT moves.
# There they acquire left dependents using LEFT moves.  After that a
# single SHIFT moves them to the top of the stack (s0).  There they
# acquire right dependents with SHIFT-RIGHT pairs.  Finally from s0
# they acquire a head with a LEFT or RIGHT move.  Any token from the
# buffer may become the right head but only s1 from the stack may
# become a left head.  The parser terminates with a single word at s0
# whose head is ROOT (represented as head=deprel=0).
#
# 1. SHIFT moves n0 to s0: n0 cannot acquire left dependents after a
# shift.  Also it can no longer get a head from the stack to the left
# of s0 or get a root head if there is s0: (0+s\s0,n0) + (n0,s)
#
# 2. RIGHT adds (s1,s0): s0 cannot acquire a head or dependent from
# the buffer after right: (s0,b) + (b,s0)
#
# 3. LEFT adds (n0,s0): s0 cannot acquire s1 or 0 (if there is no s1)
# or ni (i>0) as head.  It also cannot acquire any more right
# children: (s0,b) + (b\n0,s0) + (s1 or 0,s0)

function movecosts(p::ArcHybrid, head::AbstractArray, deprel::AbstractArray, cost::Pvec=Array(Pval,p.nmove))
    @assert (length(head) == p.nword)
    @assert (length(deprel) == p.nword)
    @assert (length(cost) == p.nmove)
    fill!(cost, Pinf)
    s = p.state
    n0 = s.wptr                                                 # n0 is top of buffer
    if (n0 <= p.nword)                                          # if n0 shift is legal moving n0 to s0
        n0h = head[n0]                                          # n0h is the actual head of n0
        SHIFT = p.nmove
        cost[SHIFT] = (sum(head[s.stack[1:s.sptr]] .== n0) +	# no more left dependents for n0
                       sum(s.stack[1:s.sptr-1] .== n0h) +       # no heads to the left of s0 for n0
                       ((n0h == 0) && (s.sptr >= 1)))           # no root head for n0 if there is s0
    end
    if (s.sptr >= 1)                                            # left/right valid if stack nonempty
        s0 = p.stack[s.sptr]                                    # s0 is top of stack
        s0h = head[s0]                                          # s0h is the actual head of s0
        s0b = sum(head[n0:end] .== s0)                          # num buffer words whose head is s0

        if (n0 <= p.nword)                                      # if n0 left is legal, making n0 head of s0
            leftcost = (s0b +                                   # no more right children for s0
                        ((s0h > n0) ||                          # no heads to the right of n0 for s0
                         ((s.sptr == 1) && (s0h == 0)) ||       # no root head for s0 if alone
                         ((s.sptr > 1) &&                       # no more s1 for head of s0
                          (s0h == s.stack[s.sptr-1]))))
            LMOVES = (2-LEFT):2:(p.nmove-1-LEFT)
            if (s0h != n0) 
                cost[LMOVES] = leftcost                         # if n0 is not the actual head we are done
            else
                cost[LMOVES] = leftcost + 1                     # otherwise +1 for moves with wrong labels
                cost[midx(LEFT,deprel[s0])] -= 1                # except for the correct label
            end                                                 # 2deprel+1 is the left move with deprel
        end
        if (s.sptr >= 2)                                        # right is legal making s1 head of s0
            s1 = s.stack[s.sptr-1]                              # s1 is the stack element before s0
            rightcost = s0b + (s0h >= n0)                       # no more right head or dependent for s0
            RMOVES = (2-RIGHT):2:(p.nmove-1-RIGHT)
            if (s0h != s1)                                      
                cost[RMOVES] = rightcost                        # if s1 is not the actual head we are done
            else
                cost[RMOVES] = rightcost + 1                    # otherwise +1 for the wrong labels
                cost[midx(RIGHT,deprel[s0])] -= 1               # except for the correct label
            end
        end
    end
    return cost
end # movecosts

anyvalidmoves(p::ArcHybrid)=((p.state.wptr <= p.nword) || (p.state.sptr >= 2))
