# features() returns a feature vector given a parser, a sentence,
# and a FeatureList.  An optional last argument can provide the
# preallocated output vector which should have length flen().
#
# A FeatureList is a ASCIIString array specifying a set of features.
# Each string specifies a particular feature and has the form:
#   [sn]\d?([hlr]\d?)*[wpdLabAB]
# 
# The meaning of each letter is below.  In the following i is a single
# digit integer which is optional if the default value is used:
#
# si: i'th stack word, default i=0 means top
# ni: i'th buffer word, default i=0 means first
# hi: i'th degree head, default i=1 means direct head
# li: i'th leftmost child, default i=1 means the leftmost child 
# ri: i'th rightmost child, default i=1 means the rightmost child
# w: word vector
# p: context vector
# d: distance to the right.  e.g. s1d is s0s1 distance, s0d is s0n0
#    distance.  encoding: 1,2,3,4,5-10,10+ (from ZN11)
# L: dependency label (0 is ROOT or NONE)
# a: number of left children.  encoding: 0,1,...,8,9+
# b: number of right children.  encoding: 0,1,...,8,9+
# A: set of left dependency labels
# B: set of right dependency labels

typealias Feature ASCIIString
typealias Fvec Vector{Feature}

function features(p::Parser, s::Sentence, feats::Fvec,
                  x::AbstractArray=Array(wtype(s),flen(p,s,feats),1), 
                  xcol::Integer=1)
    wrows = wdim(s)             # first half word, second half context
    xrows = size(x, 1)
    x0 = zero(eltype(x))
    x1 = one(eltype(x))
    x[:,xcol] = x0
    nx = 0                      # last entry in x
    nw = wrows >> 1
    nd = p.ndeps
    for f in feats
        @assert in(f[1], "sn") "feature string should start with [sn]"
        (i,n) = isdigit(f[2]) ? (f[2] - '0', 3) : (0, 2)
        (a,d) = (0,0)           # target word index and right distance
        if ((f[1] == 's') && (p.sptr - i >= 1))
            a = p.stack[p.sptr - i]
            d = (i>0 ? (p.stack[p.sptr - i + 1] - a) : 
                 p.wptr <= p.nword ? (p.wptr - a) : 0)
        elseif ((f[1] == 'n') && (p.wptr + i <= p.nword))
            a = p.wptr + i
        end
        while (fn=f[n];in(fn, "hlr"))
            d = 0
            (i,n) = isdigit(f[n+1]) ? (f[n+1] - '0', n+2) : (1, n+1)
            @assert i > 0 "hlr indexing is one based"
            a == 0 && continue
            if fn == 'l'
                @assert (a <= p.wptr) "buffer words other than n0 do not have ldeps"
                j = p.lcnt[a] - i + 1 # leftmost child at highest index
                a = (j > 0) ? p.ldep[a,j] : 0
            elseif fn == 'r'
                @assert (a < p.wptr) "buffer words do not have rdeps"
                j = p.rcnt[a] - i + 1
                a = (j > 0) ? p.rdep[a,j] : 0
            else # if fn == 'h'
                for j=1:i
                    a = p.head[a]
                    a == 0 && break
                end
            end
        end
        @assert n == length(f)
        fn = f[n]
        if (a > 0)
            if fn == 'w'
                copy!(x, (xcol-1)*xrows+nx+1, s.wvec, (a-1)*wrows+1, nw)
            elseif fn == 'p'
                copy!(x, (xcol-1)*xrows+nx+1, s.wvec, (a-1)*wrows+nw+1, nw)
            elseif fn == 'd'
                (d > 0) && (x[nx+(d>10?6:d>5?5:d), xcol] = x1)
            elseif fn == 'L'
                x[nx+1+p.deprel[a], xcol] = x1
            elseif fn == 'a'
                x[nx+1+(p.lcnt[a]>9?9:p.lcnt[a]), xcol] = x1
            elseif fn == 'b'
                x[nx+1+(p.rcnt[a]>9?9:p.rcnt[a]), xcol] = x1
            elseif fn == 'A'
                for j=1:p.lcnt[a]; x[nx+p.deprel[p.ldep[a,j]], xcol] = x1; end
            elseif fn == 'B'
                for j=1:p.rcnt[a]; x[nx+p.deprel[p.rdep[a,j]], xcol] = x1; end
            else
                error("Unknown feature $(fn)")
            end
        end # if (a > 0)
        nx += flen1(fn, nw, nd)
    end
    @assert nx == xrows
    return x
end

function flen(p::Parser, s::Sentence, feats::Fvec)
    nx = 0
    nw = wdim(s) >> 1
    nd = p.ndeps
    for f in feats
        nx += flen1(f[end], nw, nd)
    end
    return nx
end

function flen1(c::Char, nw::Integer,nd::Integer)
    c == 'w' && return nw
    c == 'p' && return nw
    c == 'd' && return 6
    c == 'L' && return(nd+1)
    c == 'a' && return 10
    c == 'b' && return 10
    c == 'A' && return nd
    c == 'B' && return nd
    error("Unknown feature character $c")
end

xsize(p::Parser, s::Sentence, f::Fvec)=(flen(p,s,f),nmoves(p,s))
xsize(p::Parser, c::Corpus, f::Fvec)=(flen(p,c[1],f),nmoves(p,c))
ysize(p::Parser, s::Sentence)=(nmoves(p),nmoves(p,s))
ysize(p::Parser, c::Corpus)=(nmoves(p),nmoves(p,c))
