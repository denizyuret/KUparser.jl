# The greedy transition parser parses the sentence using the
# following: (n::Net, p::Parser, s::Sentence)
#
# while any(v = valid(p))
#   f = features(p, s)
#   y = forw(n, f)
#   y[~v] = -inf
#   move(p, findmax(y)[2])
#
# There are two opportunities for parallelism:
# 1. We process multiple sentences to minibatch net input.
#    This speeds up forw.
# 2. We do multiple batches in parallel to utilize CPU cores.
#    This speeds up features.
#
# nworkers() gives the number of processes available

using KUnet  # ?? @everywhere using KUnet
typealias Corpus Vector{Sentence}

function gparse(corpus::Corpus, net::Net; batch=128)
    p = @parallel (vcat) for b=1:batch:length(corpus)
        e = b + batch - 1
        e > length(corpus) && (e = length(corpus))
        gparse(corpus, net, b, e)
    end
end

function gparse(corpus::Corpus, net::Net, b::Integer, e::Integer)
    
end


# what if net does not get copied?
# we may overwrite fields?
# ideal would be cpu/net copied, gpu left alone
# what if net does get copied and we run out of memory


# maybe first debug the simple parser, then parallelize...
