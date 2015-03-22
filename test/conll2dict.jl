using HDF5,JLD
const ROOT="ROOT"

# Construct mapping from strings to positive integer values
typealias StrSet Set{UTF8String}
form_hash = StrSet()
postag_hash = StrSet()
deprel_hash = StrSet()

for l in eachline(STDIN)
    l == "\n" && continue
    (form, postag, head, deprel, wvec) = split(chomp(l), '\t')
    push!(form_hash, form)
    push!(postag_hash, postag)
    deprel != ROOT && push!(deprel_hash, deprel)
end

save(ARGS[1],
     "form", sort(collect(form_hash)),
     "postag", sort(collect(postag_hash)),
     "deprel", sort(collect(deprel_hash)))
