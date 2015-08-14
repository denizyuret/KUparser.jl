using KUnet

# findprev does not exist in v0.3
if VERSION < v"0.4-"
function findprev(A, v, start)
    for i = start:-1:1
        A[i] == v && return i
    end
    0
end
end

# This is "distribute" with an extra procs argument:
# will be fixed in Julia 0.4
VERSION >= v"0.4-" && eval(Expr(:using,:DistributedArrays))
function distproc(a::AbstractArray, procs)
    owner = myid()
    rr = RemoteRef()
    put!(rr, a)
    d = DArray(size(a), procs) do I
        remotecall_fetch(owner, ()->fetch(rr)[I...])
    end
    # Ensure that all workers have fetched their localparts.
    # Else a gc in between can recover the RemoteRef rr
    for chunk in d.chunks
        wait(chunk)
    end
    d
end

# This is "sortperm!" with AbstractVector argument.
# this has been fixed in Julia 0.4:
# VERSION >= v"0.4.0-" && eval(Expr(:import,:Base,:sortperm!))
using Base: Algorithm, Ordering, DEFAULT_UNSTABLE, Forward, Perm, ord
function sortpermx{I<:Integer}(x::AbstractVector{I}, v::AbstractVector; alg::Algorithm=DEFAULT_UNSTABLE,
                               lt::Function=isless, by::Function=identity, rev::Bool=false, order::Ordering=Forward,
                               initialized::Bool=false)
    length(x) != length(v) && throw(ArgumentError("Index vector must be the same length as the source vector."))
    !initialized && @inbounds for i = 1:length(v); x[i] = i; end
    sort!(x, alg, Perm(ord(lt,by,rev,order),v))
end

# Compute relevant scores
function evalparse(parsers, sentences; ispunct=zn11punct, postag=[])
    ph = map(p->p.head, parsers)
    sh = map(s->s.head, sentences)
    pd = map(p->p.deprel, parsers)
    sd = map(s->s.deprel, sentences)
    wf = map(s->s.form, sentences)
    sp = map(s->s.postag, sentences)
    phcat = vcat2(ph)
    shcat = vcat2(sh)
    pdcat = vcat2(pd)
    sdcat = vcat2(sd)
    wfcat = vcat2(wf)
    spcat = vcat2(sp)
    uas = mean(phcat .== shcat) # unlabeled attachment score
    las = mean((phcat .== shcat) & (pdcat .== sdcat)) # labeled attachment score
    uem = mean(ph .== sh)                             # unlabeled exact match
    lem = mean((ph .== sh) & (pd .== sd))             # labeled exact match
    if ispunct == kcc08punct
        @assert !isempty(postag)
        isword = map(p->!ispunct(p), postag[spcat])
    else
        isword = map(w->!ispunct(w), wfcat)
    end
    uas2 = mean(phcat[isword] .== shcat[isword])
    las2 = mean((phcat[isword] .== shcat[isword]) & (pdcat[isword] .== sdcat[isword]))
    return (uas, uas2, las, las2, uem, lem)
end

function vcat2{T}(v::Vector{Vector{T}})
    w = similar(v[1], 0)
    for vi in v; append!(w, vi); end
    return w
end

# Everybody means something else by "excluding punctuation":
# The buggy conll07 eval.pl script looks for non-alpha characters, but excludes ` and $, also won't match -LRB-, -RRB- etc.
c07punct(w)=(ismatch(r"^\W+$",w) && !ismatch(r"^[`$]+$",w))
# ZN11 eval.py uses his own regexp for punctuation (I think both for chinese and english)
zn11punct(w)=ismatch(r"^[,?!:;]$|^-LRB-$|^-RRB-$|^[.]+$|^[`]+$|^[']+$|^（$|^）$|^、$|^。$|^！$|^？$|^…$|^，$|^；$|^／$|^：$|^“$|^”$|^「$|^」$|^『$|^』$|^《$|^》$|^一一$",w)
# KCC08 uses postags rather than wordforms, but for some reason skips #, $, -LRB-, -RRB- etc.
kcc08punct(p)=in(p,["``", "''", ":", ",", "."])

# Copying net between cpu and gpu

import Base: copy
KUnet.GPU && eval(Expr(:using,:CUDArt))

function copy(net::Net, to::Symbol)
    net = KUnet.strip!(net)
    ((to == :gpu) ? gpucopy(net) :
     (to == :cpu) ? cpucopy(net) :
     error("Don't know how to copy net to $to"))
end

function testnet(net)
    net = cpucopy(net)
    KUnet.strip!(net)
    for l in net
        for n in fieldnames(l)
            if isdefined(l,n) && isa(l.(n), KUparam)
                # Get rid of all the training fields
                l.(n) = KUparam(l.(n).arr)
            end
        end
    end
    return net
end
