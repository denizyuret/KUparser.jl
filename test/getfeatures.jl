# args: conll-file-prefix feature-name parser-type
using HDF5,JLD,KUparser
costr = length(ARGS) >= 1 ? "$(ARGS[1])" : "acl11"
ftstr = length(ARGS) >= 2 ? "$(ARGS[2])" : "zn11orig"
ptstr = length(ARGS) >= 3 ? "$(ARGS[3])" : "ArcEager13"
@show (costr, ftstr, ptstr)
ft = eval(parse("Flist.$ftstr"))
pt = eval(parse(ptstr))
@date dev=load("$costr.dev.jld4") # corpus, deprel, postag
@date tst=load("$costr.tst.jld4")
@date trn=load("$costr.trn.jld4")
@assert trn["deprel"] == dev["deprel"] == tst["deprel"]
@assert trn["postag"] == dev["postag"] == tst["postag"]
nd = length(dev["deprel"])
@date (pdev,xdev,ydev) = oparse(pt, dev["corpus"], nd, ft)
@date (ptst,xtst,ytst) = oparse(pt, tst["corpus"], nd, ft)
@date (ptrn,xtrn,ytrn) = oparse(pt, trn["corpus"], nd, ft)
if isa(xtrn, SparseMatrixCSC)
    nrows = max(maximum(xtrn.rowval), maximum(xtst.rowval), maximum(xdev.rowval))
    xtrn.m = xtst.m = xdev.m = nrows
end
@date @save "$costr-$ftstr.jld" xtrn ytrn xdev ydev xtst ytst 
