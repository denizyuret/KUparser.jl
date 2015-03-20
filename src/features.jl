# f::Features is a nx3 matrix whose rows determine which features to extract
# Each row of f consists of the following three values:
# 1. anchor word: 0:n0, 1:n1, 2:n2, ..., -1:s0, -2:s1, -3:s2, ...
# 2. target word: 0:self, 1:rightmost child, 2:second rightmost, -1:leftmost, -2:second leftmost ...
# 3. feature: One of the features listed below.
#
# 0:wvec (word+context if token, dim depends on encoding)
# +-1: exists/doesn't (one bit)
# +-2: right/left child count (4 bits, == encoding: 0,1,2,3+)
# +3: distance to right (4 bits, == encoding: 1,2,3,4+, root is 4+)
# -3: average of in-between tokens to the right (dim:wvec)
# +-4: word/context half of vector (dim:wvec/2, only valid for token encoding, assumes first half=word, second half=context)
# +-5: right/left child count (4 bits, >= encoding: >=1, >=2, >=3, >=4)
# +-6: right/left child count (4 bits, <= encoding: <=0, <=1, <=2, <=3)
# +7: distance to right, >= encoding (8 bits, >= encoding: >=2, >=3, >=4, >=6, >=8, >=12, >=16, >=20)
# -7: distance to right, <= encoding (8 bits, <= encoding: <=1, <=2, <=3, <=4, <=6, <=8, <=12, <=16)
# +-8: average of in-between word/context vectors to the right (dim:wvec/2)
# +-9: head exists/doesn't (one bit)

const flength = Int[1, -1, 8, 4, 4, -1, -2, 4, 1, -2, 1, 4, 4, -1, 4, 4, 8, -1, 1]
const foffset = 10

function flen(ndim::Int, f::Features)
    nfeat = size(f, 1)
    ndim2 = ndim>>1
    len = 0;
    for i=1:nfeat
        l = flength[f[i,3] + foffset]
        len += (l > 0 ? l : -l*ndim2)
    end
    return len
end

# fidx[ifeat]: end of feature f[ifeat,:] in x

function fidx(ndim::Int, f::Features)
    nfeat = size(f, 1)
    ndim2 = ndim>>1
    idx = Array(Int, nfeat)
    for i=1:nfeat
        l = flength[f[i,3] + foffset]
        idx[i] = (i > 1 ? idx[i-1] : 0) + (l > 0 ? l : -l*ndim2)
    end
    return idx
end

function features(p::Parser, s::Sentence, f::Features)
    x = Array(eltype(s.wvec), flen(size(s.wvec,1), f), 1)
    features(p, s, f, x)
end

function features(p::Parser, s::Sentence, f::Features, x)

    # Given a parser state p, a sentence s, and a feature matrix f, fills and
    # returns a feature vector x, which should be preallocated to have
    # flen(size(s.wvec,1), f) dimensions.

    ndim = size(s.wvec,1)               # token vector dimensionality
    ndim2 = ndim>>1                     # for token encodings the first half is the word vector, the second half is the context vector
    nfeat = size(f, 1)                  # number of features
    smax = Pinf                         # maximum distance in sentence
    i = 0                               # i: index into feature vector x
    fill!(x, zero(eltype(x)))

    for j=1:nfeat               # j: index into feature matrix f

	# identify the anchor a: 8.81us
        if f[j,1] >= 0                       # buffer word
            a = p.wptr + f[j,1]
            (a > p.nword) && (a = 0)
        else                                  # stack word
            ax = p.sptr + f[j,1] + 1
            if (ax > 0) 
                a = p.stack[ax]
                @assert ((a >= 1) && (a < p.wptr)) "Bad anchor"
            else 
                a = 0 
	    end
	end

	# identify the target b: 3.13us
	if a == 0
	    b = 0
	elseif f[j,2] == 0                   # self
	    b = a                              
	elseif f[j,2] > 0                    # right-child
	    @assert (a < p.wptr) "buffer words do not have rdeps"
	    nc = p.rcnt[a]
	    bx = nc - f[j,2] + 1
	    if (bx > 0)
		b = p.rdep[a, bx]
		@assert ((b > a) && (b <= p.nword) && (p.head[b] == a)) "Bad rdep"
	    else
		b = 0
	    end
	elseif f[j,2] < 0                    # left-child
	    @assert (a <= p.wptr) "buffer words other than n0 do not have ldeps"
	    nc = p.lcnt[a]
	    bx = nc + f[j,2] + 1
	    if (bx > 0)
		b = p.ldep[a, bx]
		@assert ((b >= 1) && (b < a) && (p.head[b] == a)) "Bad ldep"
	    else
		b = 0
	    end
	end

	# generate the feature: 4.16us
	if f[j,3] == 0	# wvec
	    (b > 0) && (x[i+1:i+ndim] = s.wvec[:,b])
	    i+=ndim; 

	elseif f[j,3] == 1       # exists
	    (b > 0) && (x[i+1] = 1)
	    i+=1; 

	elseif f[j,3] == -1      # does not exist
	    (b == 0) && (x[i+1] = 1)
	    i+=1; 

	elseif f[j,3] == 2       # rdep count, 4 bits, == encoding
	    if (b > 0) 
		@assert (b < p.wptr) "buffer words do not have rdeps"
		nc = p.rcnt[b]
		(nc > 3) && (nc = 3)
		x[i+1+nc] = 1
	    end 
	    i+=4; 

	elseif f[j,3] == -2      # ldep count, 4 bits, == encoding
	    if (b > 0)
		@assert (b <= p.wptr) "buffer words other than n0 do not have ldeps"
		nc = p.lcnt[b] 
		(nc > 3) && (nc = 3)
		x[i+1+nc] = 1
	    end 
	    i+=4; 

	elseif f[j,3] == 3       # distance to the right
	    @assert (f[j,1] < 0 && f[j,2] == 0) "distance only available for stack words"
	    if (b > 0)
		if f[j,1] == -1 # s0n0 distance
		    c = (p.wptr <= p.nword ? p.wptr : b+smax) # root is far far away...
		else      # s(i)-s(i-1) distance
		    cx = p.sptr + f[j,1] + 2
		    c = p.stack[cx]
		end
		@assert (c > b) "c <= b"
		d = c - b
		(d > 4) && (d = 4)
		x[i+d] = 1
	    end 
	    i+=4; 

	elseif f[j,3] == 4       # word (first) half of token vector
	    (b > 0) && (for d=1:ndim2; x[i+d]=s.wvec[d,b]; end)
	    i+=ndim2; 

	elseif f[j,3] == -4      # context (second) half of token vector
	    (b > 0) && (for d=1:ndim2; x[i+d]=s.wvec[ndim2+d,b]; end)
	    i+=ndim2; 

	elseif f[j,3] == 5       # rdep count, 4 bits, >= encoding
	    if (b > 0) 
		@assert (b < p.wptr) "buffer words do not have rdeps"
		nc = p.rcnt[b]
		for ic=1:4; (nc >= ic) && (x[i+ic] = 1); end
	    end 
	    i+=4; 

	elseif f[j,3] == -5      # ldep count, 4 bits, >= encoding
	    if (b > 0)
		@assert (b <= p.wptr) "buffer words other than n0 do not have ldeps"
		nc = p.lcnt[b]
		for ic=1:4; (nc >= ic) && (x[i+ic] = 1); end
	    end 
	    i+=4; 

	elseif f[j,3] == 6       # rdep count, 4 bits, <= encoding
	    if (b > 0) 
		@assert (b < p.wptr) "buffer words do not have rdeps"
		nc = p.rcnt[b]
		for ic=0:3; (nc <= ic) && (x[i+1+ic] = 1); end
	    end 
	    i+=4; 

	elseif f[j,3] == -6      # ldep count, 4 bits, <= encoding
	    if (b > 0)
		@assert (b <= p.wptr) "buffer words other than n0 do not have ldeps"
		nc = p.lcnt[b]
		for ic=0:3; (nc <= ic) && (x[i+1+ic] = 1); end
	    end 
	    i+=4; 

	elseif f[j,3] == 7       # distance to the right, >= encoding, 8 bits
	    @assert (f[j,1] < 0 && f[j,2] == 0) "distance only available for stack words"
	    if (b > 0)
		if f[j,1] == -1 # s0n0 distance
		    c = (p.wptr <= p.nword ? p.wptr : b+smax) # root is far far away...
		else      # s(i)-s(i-1) distance
		    cx = p.sptr + f[j,1] + 2
		    c = p.stack[cx]
		end
		@assert (c > b) "c <= b"
		d = c - b
		dmin = [2,3,4,6,8,12,16,20]
		for id=1:length(dmin); (d >= dmin[id]) && (x[i+id] = 1); end
	    end 
	    i+=8; 

	elseif f[j,3] == -7       # distance to the right, <= encoding, 8 bits
	    @assert (f[j,1] < 0 && f[j,2] == 0) "distance only available for stack words"
	    if (b > 0)
		if f[j,1] == -1 # s0n0 distance
		    c = (p.wptr <= p.nword ? p.wptr : b+smax) # root is far far away...
		else      # s(i)-s(i-1) distance
		    cx = p.sptr + f[j,1] + 2
		    c = p.stack[cx]
		end
		@assert (c > b) "c <= b"
		d = c - b
		dmax = [1,2,3,4,6,8,12,16]
		for id=1:length(dmax); (d <= dmax[id]) && (x[i+id] = 1); end
	    end 
	    i+=8; 

	elseif f[j,3] == -3      # avg of in-between token vectors to the right
	    @assert (f[j,1] < 0 && f[j,2] == 0) "in-between only available for stack words"
	    if (b > 0)
		if f[j,1] == -1 # s0n0 interval
		    c = (p.wptr <= p.nword ? p.wptr : b) # no in-between with root
		else      # s(i)-s(i-1) interval
		    cx = p.sptr + f[j,1] + 2
		    c = p.stack[cx]
		end
		if c > b+1
                    for d=1:ndim
                        for bc=(b+1):(c-1)
                            x[i+d] += s.wvec[d,bc]
                        end
                        x[i+d] /= (c-b-1)
                    end
                end
	    end
	    i+=ndim; 
	    
	elseif f[j,3] == 8      # avg of in-between word vectors to the right
	    @assert (f[j,1] < 0 && f[j,2] == 0) "in-between only available for stack words"
	    if (b > 0)
		if f[j,1] == -1 # s0n0 interval
		    c = (p.wptr <= p.nword ? p.wptr : b) # no in-between with root
		else      # s(i)-s(i-1) interval
		    cx = p.sptr + f[j,1] + 2
		    c = p.stack[cx]
		end
		if c > b+1
                    for d=1:ndim2
                        for bc=(b+1):(c-1)
                            x[i+d] += s.wvec[d,bc]
                        end
                        x[i+d] /= (c-b-1)
                    end
		end
	    end
	    i+=ndim2; 

	elseif f[j,3] == -8      # avg of in-between context vectors to the right
	    @assert (f[j,1] < 0 && f[j,2] == 0) "in-between only available for stack words"
            if (b > 0)
                if f[j,1] == -1 # s0n0 interval
                    c = (p.wptr <= p.nword ? p.wptr : b) # no in-between with root
                else      # s(i)-s(i-1) interval
                    cx = p.sptr + f[j,1] + 2
                    c = p.stack[cx]
                end
                if c > b+1
                    for d=1:ndim2
                        for bc=(b+1):(c-1)
                            x[i+d] += s.wvec[ndim2+d,bc]
                        end
                        x[i+d] /= (c-b-1)
                    end
                end
            end
            i+=ndim2; 

	elseif f[j,3] == 9       # head exists
	    (b > 0) && (x[i+1] = (p.head[b] > 0))
	    i+=1; 

	elseif f[j,3] == -9      # head does not exist
	    (b > 0) && (x[i+1] = (p.head[b] == 0))
	    i+=1; 

	else
	    error("Unknown feature $(f[j,3])")
        end # if f[j,3]...
    end # for j=1:nfeat
    return x
end # features
