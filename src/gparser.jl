# ?? @everywhere using KUnet

# The greedy transition parser parses the sentence using the
# following:

function gparse(s::Sentence, n::Net, f::Fmat)
    p = ArcHybrid(size(s.wvec,2))
    while (v = valid(p); any(v))
        x = features(p, s, f)
        #y = predict(n, x)[:]
        y = rand(eltype(x), NMOVE)
        y[!v] = -Inf
        move!(p, indmax(y))
    end
    p.head
end

function gparse(c::Corpus, n::Net, f::Fmat)
    map(s->gparse(s,n,f), c)
end

# There are two opportunities for parallelism:
# 1. We process multiple sentences to minibatch net input.
#    This speeds up forw.
# 2. We do multiple batches in parallel to utilize CPU cores.
#    This speeds up features.
#
# nworkers() gives the number of processes available

# function gparse(corpus::Corpus, net::Net; batch=128)
#     p = @parallel (vcat) for b=1:batch:length(corpus)
#         e = b + batch - 1
#         e > length(corpus) && (e = length(corpus))
#         gparse(corpus, net, b, e)
#     end
# end

# function gparse(corpus::Corpus, net::Net, b::Integer, e::Integer)
    
# end


# what if net does not get copied?
# we may overwrite fields?
# ideal would be cpu/net copied, gpu left alone
# what if net does get copied and we run out of memory


# maybe first debug the simple parser, then parallelize...
