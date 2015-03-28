using CUDArt
using KUnet

# Print date, expression and elapsed time after execution
VERSION < v"0.4-" && eval(Expr(:using,:Dates))
macro date(_x) :(println("$(now()) "*$(string(_x)));flush(STDOUT);@time $(esc(_x))) end

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

# Only copy what is needed for predict
function testnet(l::KUnet.Layer)
    ll = KUnet.Layer()
    isdefined(l,:w) && (ll.w = to_host(l.w))
    isdefined(l,:b) && (ll.b = to_host(l.b))
    isdefined(l,:f) && (ll.f = l.f)
    return ll
end

testnet(net::KUnet.Net)=map(l->testnet(l), net)


# Compute relevant scores
function evalparse(parsers, sentences)
    ph = map(p->p.head, parsers)
    sh = map(s->s.head, sentences)
    pd = map(p->p.deprel, parsers)
    sd = map(s->s.deprel, sentences)
    wf = map(s->s.form, sentences)
    phcat = vcat(ph...)
    shcat = vcat(sh...)
    pdcat = vcat(pd...)
    sdcat = vcat(sd...)
    wfcat = vcat(wf...)
    uas = mean(phcat .== shcat)
    las = mean((phcat .== shcat) & (pdcat .== sdcat))
    uem = mean(ph .== sh)
    lem = mean((ph .== sh) & (pd .== sd))
    isword = map(w->(!ismatch(r"^\W+$",w) || ismatch(r"^[`$]+$",w)), wfcat)
    uas2 = mean(phcat[isword] .== shcat[isword])
    las2 = mean((phcat[isword] .== shcat[isword]) & (pdcat[isword] .== sdcat[isword]))
    return (uas, uas2, las, las2, uem, lem)
end

