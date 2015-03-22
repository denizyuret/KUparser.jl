using HDF5,JLD,DataStructures
using KUparser: Sentence, Pval, Dval

# Store strings as positive integer values
form_hash = OrderedDict{UTF8String,UInt32}()
postag_hash = OrderedDict{UTF8String,Dval}()
deprel_hash = OrderedDict{UTF8String,Dval}()
getid(h,x)=get!(h,convert(UTF8String,x),1+length(h))

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
    form = getid(form_hash, form)
    postag = getid(postag_hash, postag)
    deprel = getid(deprel_hash, deprel)
    head = convert(Pval, int(head))
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
save(ARGS[2], "corpus", corpus,
     "form", collect(keys(form_hash)),
     "postag", collect(keys(postag_hash)),
     "deprel", collect(keys(deprel_hash)))
