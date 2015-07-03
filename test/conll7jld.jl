# args: dictfile outfile < input

using HDF5,JLD # ,DataStructures
using KUparser: Sentence, Position, DepRel

# Store postag/deprel strings as positive integer values
postag_hash = Dict{UTF8String,DepRel}()
deprel_hash = Dict{UTF8String,DepRel}()

# First argument is a dictionary file, could be an existing jld file
dict = load(ARGS[1])
for i=1:length(dict["postag"]); postag_hash[dict["postag"][i]] = i; end
for i=1:length(dict["deprel"]); deprel_hash[dict["deprel"][i]] = i; end

# ROOT deprel is stored as the special value 0
const ROOT="ROOT" 
deprel_hash[ROOT] = 0

# The input is in conll07 format followed by wvec+cvec and read from stdin:
# ID,FORM,LEMMA,CPOSTAG,POSTAG,FEATS,HEAD,DEPREL,PHEAD,PDEPREL

corpus  = Sentence[]
s = nothing
mydims = 0
for l in eachline(STDIN)
    if l == "\n"
        @assert s != nothing
        push!(corpus, s)
        s = nothing
        continue
    end
    (id, form, lemma, cpostag, postag, feats, head, deprel, phead, pdeprel, wvec) = split(chomp(l), '\t')
    form = convert(UTF8String, form)
    postag = postag_hash[postag]
    deprel = deprel_hash[deprel]
    head = convert(Position, int(head))
    wvec = map(float32, split(wvec, ' '))
    if s == nothing
        mydims = length(wvec)
        s = Sentence()
        s.form = [form]
        s.postag = [postag]
        s.head = [head]
        s.deprel = [deprel]
        s.wvec = wvec''
    else
        @assert mydims == length(wvec)
        push!(s.form, form)
        push!(s.postag, postag)
        push!(s.head, head)
        push!(s.deprel, deprel)
        s.wvec = hcat(s.wvec, wvec)
    end
end

@assert s == nothing
@assert length(corpus) > 0
save(ARGS[2], "corpus", corpus, "postag", dict["postag"], "deprel", dict["deprel"])
