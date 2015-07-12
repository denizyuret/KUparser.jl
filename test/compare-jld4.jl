# compare two jld4 files

using HDF5,JLD,KUparser
a=load(ARGS[1])
b=load(ARGS[2])
@show a["deprel"]==b["deprel"]
@show a["postag"]==b["postag"]
aform=map(x->x.form, a["corpus"]); bform=map(x->x.form, b["corpus"])
@show aform==bform
apostag=map(x->x.postag, a["corpus"]); bpostag=map(x->x.postag, b["corpus"])
@show apostag==bpostag
ahead=map(x->x.head, a["corpus"]); bhead=map(x->x.head, b["corpus"])
@show ahead==bhead
adeprel=map(x->x.deprel, a["corpus"]); bdeprel=map(x->x.deprel, b["corpus"])
@show adeprel==bdeprel
awvec=(isdefined(a["corpus"][1],:wvec) ? map(x->x.wvec, a["corpus"]) : Any[])
bwvec=(isdefined(b["corpus"][1],:wvec) ? map(x->x.wvec, b["corpus"]) : Any[])
@show awvec==bwvec
