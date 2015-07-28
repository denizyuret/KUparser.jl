# features(p,s,f,x,n) returns a feature vector given a parser, a
# sentence, and an Fvec (feature specification vector, explained
# below).  The last two arguments (optional) can provide a
# preallocated output matrix which should have height flen(), and the
# column number in that matrix to be filled.
#
# Features can be dense or sparse.  Dense features are specified by
# string names:

typealias DFeature ASCIIString

# Each string specifies a particular feature and has the form:
#   [sn]\d?([hlr]\d?)*[vcpdLabAB]
# 
# The meaning of each letter is below.  In the following i is a single
# digit integer which is optional if the default value is used:
#
# si: i'th stack word, default i=0 means top
# ni: i'th buffer word, default i=0 means first
# hi: i'th degree head, default i=1 means direct head
# li: i'th leftmost child, default i=1 means the leftmost child 
# ri: i'th rightmost child, default i=1 means the rightmost child
# v: word vector
# c: context vector
# p: postag
# d: distance to the right.  e.g. s1d is s0s1 distance, s0d is s0n0
#    distance.  encoding: 1,2,3,4,5-10,10+ (from ZN11)
# L: dependency label (0 is ROOT or NONE)
# a: number of left children.  encoding: 0,1,...,8,9+
# b: number of right children.  encoding: 0,1,...,8,9+
# A: set of left dependency labels
# B: set of right dependency labels

# A DFvec is a String vector specifying a set of dense features.

typealias DFvec Vector{ASCIIString}

# Fvec is a union of DFvec and SFvec, which specify dense and sparse
# features respectively.  The user controls what kind of features are
# used by using the appropriate type of feature vector.

typealias Fvec Union(DFvec,SFvec)

function features(p::Parser, s::Sentence, feats::DFvec, #
                  x::AbstractArray=Array(wtype(s),flen(p,s,feats),1), 
                  xcol::Integer=1)
    xcol <= size(x,2) || error("xcol > size(x,2)")
    wrows = wdim(s)             # first half word, second half context
    xrows = size(x,1)
    xtype = eltype(x)
    x1 = one(xtype)
    x[:,xcol] = zero(xtype)     # 366
    nx = 0                      # last entry in x
    nv = wrows >> 1             # size of word/context vector, assumes word vec and context vec concatenated
    nd = p.ndeps
    np = 45                     # hardcoding the ptb postag count for now
    nw = p.nword

    ldep = Array(Vector{Int}, nw)
    rdep = Array(Vector{Int}, nw)
    lset = zeros(xtype, nd, nw)      # 82
    rset = zeros(xtype, nd, nw)      # 980
    for d=1:nw
        h=int(p.head[d])        # 493
        if h==0
            continue
        elseif d<h
            isdefined(ldep,h) || (ldep[h]=Array(Int,0))
            push!(ldep[h],d)    # 864
            lset[p.deprel[d],h]=1 # 396
        elseif d>h
            isdefined(rdep,h) || (rdep[h]=Array(Int,0))
            unshift!(rdep[h],d) # 401
            rset[p.deprel[d],h]=1 # 6
        else
            error("h==d")
        end
    end

    for f in feats
        f1 = f[1]; f2 = f[2]
        (i,n) = isdigit(f2) ? (f2 - '0', 3) : (0, 2) # feature digit and next character
        a = d = 0           # target word index and right distance
        if (f1 == 's')
            if (p.sptr - i >= 1) # 25
                a = int(p.stack[p.sptr - i])             # 456
                d = (i>0 ? (p.stack[p.sptr - i + 1] - a) : # 263
                     p.wptr <= p.nword ? (p.wptr - a) : 0)
            end
        elseif (f1 == 'n') 
            (p.wptr + i <= p.nword) && (a = int(p.wptr + i))
        else 
            error("feature string should start with [sn]")
        end
        while n < length(f)
            f1 = f[n]; f2 = f[n+1]
            d = 0 # dist only defined for stack words
            (i,n) = isdigit(f2) ? (f2 - '0', n+2) : (1, n+1) # 112 
            i > 0 || error("hlr indexing is one based") # 3 [lrh] is one based, [sn] was zero based
            a == 0 && continue                        # 366?
            if f1 == 'l'                              # 2
                (a <= p.wptr) || error("buffer words other than n0 do not have ldeps") # 252
                a = (isdefined(ldep,a) && i <= length(ldep[a])) ? ldep[a][i] : 0 # 374
            elseif f1 == 'r'
                (a < p.wptr) || error("buffer words do not have rdeps")
                a = (isdefined(rdep,a) && i <= length(rdep[a])) ? rdep[a][i] : 0 # 272
            elseif f1 == 'h'
                for j=1:i       # 5
                    a = int(p.head[a]) # 147
                    a == 0 && break # 59
                end
            else 
                break
            end
        end
        n == length(f) || error("n!=length(f)")
        fn = f[n]               # 23
        if fn == 'v'        # 293
            (a>0) && copy!(x, (xcol-1)*xrows+nx+1, s.wvec, (a-1)*wrows+1, nv) # 314
            nx += nv
        elseif fn == 'c'    # 123
            (a>0) && copy!(x, (xcol-1)*xrows+nx+1, s.wvec, (a-1)*wrows+nv+1, nv) # 489
            nx += nv
        elseif fn == 'p'    # 4
            (a>0) && (s.postag[a] > np) && error("postag out of bound") # 147
            (a>0) && (x[nx+s.postag[a], xcol] = x1) # 126
            nx += np
        elseif fn == 'd'                 # 3
            (a>0) && (d>0) && (x[nx+(d>10?6:d>5?5:d), xcol] = x1) # 232
            nx += 6
        elseif fn == 'L'
            (a>0) && (p.deprel[a] > nd) && error("deprel out of bound") # 110
            (a>0) && (x[nx+1+p.deprel[a], xcol] = x1) # 41 first bit for deprel=0 (ROOT)
            nx += (nd+1)
        elseif fn == 'a'
            (a>0) && (lcnt=(isdefined(ldep,a) ? length(ldep[a]) : 0); x[nx+1+(lcnt>9?9:lcnt), xcol] = x1) # 155 0-9
            nx += 10
        elseif fn == 'b'
            (a>0) && (rcnt=(isdefined(rdep,a) ? length(rdep[a]) : 0); x[nx+1+(rcnt>9?9:rcnt), xcol] = x1) # 28  0-9
            nx += 10
        elseif fn == 'A'
            (a>0) && copy!(x, (xcol-1)*xrows+nx+1, lset, (a-1)*nd+1, nd) # 266
            nx += nd
        elseif fn == 'B'
            (a>0) && copy!(x, (xcol-1)*xrows+nx+1, rset, (a-1)*nd+1, nd) # 33
            nx += nd
        else
            error("Unknown feature $(fn)") # 3
        end
    end
    nx == xrows || error("Bad feature vector length")
    return x
end

function flen(p::Parser, s::Sentence, feats::DFvec)
    nx = 0
    nw = wdim(s) >> 1
    nd = p.ndeps
    for f in feats
        nx += flen1(f[end], nw, nd) # 1129
    end
    return nx
end

function flen1(c::Char, nw::Integer=100, nd::Integer=11, np::Integer=45)
    (c == 'v' ? nw :
     c == 'c' ? nw :
     c == 'p' ? np :
     c == 'd' ? 6 :
     c == 'L' ? (nd+1) :        # ROOT is not included in nd
     c == 'a' ? 10 :
     c == 'b' ? 10 :
     c == 'A' ? nd :
     c == 'B' ? nd :
     error("Unknown feature character $c"))
end

# Utility functions to calculate the size of the feature matrix

xsize(p::Parser, s::Sentence, f::Fvec)=(flen(p,s,f),nmoves(p,s))
xsize(p::Parser, c::Corpus, f::Fvec)=(flen(p,c[1],f),nmoves(p,c))
xsize{T<:Parser}(p::Vector{T}, c::Corpus, f::Fvec)=xsize(p[1],c,f)
ysize(p::Parser, s::Sentence)=(nmoves(p),nmoves(p,s))
ysize(p::Parser, c::Corpus)=(nmoves(p),nmoves(p,c))
ysize{T<:Parser}(p::Vector{T}, c::Corpus)=ysize(p[1],c)
