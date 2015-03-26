# featurevec() returns a feature vector given a parser, a sentence,
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

typealias FeatureList Vector{ASCIIString}

function featurevec(p::Parser, s::Sentence, flist::FeatureList,
                    x::AbstractArray=Array(wtype(s),flen(p,s,flist),1))
    fill!(x, zero(eltype(x)))
    nx = 0
    nw = wdim(s) >> 1
    x1 = one(eltype(x))
    for f in flist
        @assert in(f[1], "sn") "feature string should start with [sn]"
        (i,n) = isdigit(f[2]) ? (f[2] - '0', 3) : (0, 2)
        (a,d) = (0,0)           # target word index and right distance
        if ((f[1] == 's') && (p.sptr - i >= 1))
            a = p.stack[p.sptr - i]
            d = (i>0) ? (p.stack[p.sptr - i + 1] - a) : (p.wptr - a)
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
        @assert in(fn, "wpdLabAB")
        if fn == 'w'
            (a > 0) && copy!(sub(x, nx+1:(nx+nw)), sub(s.wvec, 1:nw, int(a))); nx+=nw
        elseif fn == 'p'
            (a > 0) && copy!(sub(x, nx+1:(nx+nw)), sub(s.wvec, nw+1:nw+nw, int(a))); nx+=nw
        elseif fn == 'd'
            (a > 0) && (d > 0) && (x[nx+(d>10?6:d>5?5:d)] = x1); nx+=6
        elseif fn == 'L'
            (a > 0) && (x[nx+1+p.deprel[a]] = x1); nx+=(p.ndeps+1) # 0 is ROOT/NA
        elseif fn == 'a'
            (a > 0) && (x[nx+1+(p.lcnt[a]>9?9:p.lcnt[a])] = x1); nx+=10
        elseif fn == 'b'
            (a > 0) && (x[nx+1+(p.rcnt[a]>9?9:p.rcnt[a])] = x1); nx+=10
        elseif fn == 'A'
            (a > 0) && (for j=1:p.lcnt[a]; x[nx+p.deprel[p.ldep[a,j]]] = x1; end); nx+=p.ndeps
        elseif fn == 'B'
            (a > 0) && (for j=1:p.rcnt[a]; x[nx+p.deprel[p.rdep[a,j]]] = x1; end); nx+=p.ndeps
        else
            error("Unknown feature $(fn)")
        end
    end
    @assert nx == length(x)
    return x
end

function flen(p::Parser, s::Sentence, flist::FeatureList)
    nx = 0
    nw = wdim(s) >> 1
    nd = p.ndeps
    fs = "wpdLabAB"
    dx = [nw,nw,6,nd+1,10,10,nd,nd]
    for f in flist
        i = search(fs, f[end])
        @assert i>0
        nx += dx[i]
    end
    return nx
end

