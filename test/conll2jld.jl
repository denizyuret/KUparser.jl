using HDF5,JLD,DataStructures
using KUparser: Sentence, Pval, Dval
const ROOT="ROOT"               # ROOT deprel is stored as the special value 0

# Store strings as positive integer values
form_hash = Dict{UTF8String,UInt32}()
postag_hash = Dict{UTF8String,Dval}()
deprel_hash = Dict{UTF8String,Dval}()

# First argument is a dictionary file
dict = load(ARGS[1])
for i=1:length(dict["form"]); form_hash[dict["form"][i]] = i; end
for i=1:length(dict["postag"]); postag_hash[dict["postag"][i]] = i; end
for i=1:length(dict["deprel"]); deprel_hash[dict["deprel"][i]] = i; end
deprel_hash[ROOT] = 0

# Second argument is a 4+n column conll file
f = open(ARGS[2])
corpus  = Sentence[]
s = nothing
ndims = 0
for l in eachline(f)
    if l == "\n"
        @assert s != nothing
        push!(corpus, s)
        s = nothing
        continue
    end
    (form, postag, head, deprel, wvec) = split(chomp(l), '\t')
    form = form_hash[form]
    postag = postag_hash[postag]
    deprel = deprel_hash[deprel]
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
save(ARGS[3], "corpus", corpus)
