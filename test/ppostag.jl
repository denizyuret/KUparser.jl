# make predicted postag file
# args: gold-postag file with vectors, predicted postag file without vectors, output file

using HDF5,JLD,KUparser
a=load(ARGS[1])
b=load(ARGS[2])
@show a["deprel"]==b["deprel"]
@show a["postag"]==b["postag"]
aform=map(x->x.form, a["corpus"]); bform=map(x->x.form, b["corpus"])
@show aform==bform
for i=1:length(a["corpus"])
    a["corpus"][i].postag = b["corpus"][i].postag
end
save(ARGS[3], "corpus", a["corpus"], "postag", a["postag"], "deprel", a["deprel"])
