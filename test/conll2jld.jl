using HDF5,JLD
using KUparser: Sentence, Pval
corpus  = Sentence[]
f = open(ARGS[1])
s = nothing
ndims = 0
for l in eachline(f)
    if l == "\n"
        push!(corpus, s)
        s = nothing
        continue
    end
    (form, postag, head, deprel, wvec) = split(chomp(l), '\t')
    form = convert(UTF8String, form)
    postag = convert(UTF8String, postag)
    head = convert(Pval, int(head))
    deprel = convert(UTF8String, deprel)
    wvec = map(float32, split(wvec, ' '))
    if s == nothing
        ndims = length(wvec)
        s = Sentence()
        s.form = [form]
        s.postag = [postag]
        s.head = [head]
        s.deprel = [deprel]
        s.wvec = wvec''
    else
        @assert ndims == length(wvec)
        push!(s.form, form)
        push!(s.postag, postag)
        push!(s.head, head)
        push!(s.deprel, deprel)
        s.wvec = hcat(s.wvec, wvec)
    end
end

@assert s == nothing
@assert length(corpus) > 0
save(ARGS[2], "corpus", corpus)
