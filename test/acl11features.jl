using HDF5,JLD,KUparser
ftstr = length(ARGS) >= 1 ? "$(ARGS[1])" : "zn11orig"
ptstr = length(ARGS) >= 2 ? "$(ARGS[2])" : "ArcEager13"
@show (ftstr, ptstr)
ft = eval(parse("Flist.$ftstr"))
pt = eval(parse(ptstr))
@date dev=load("acl11.dev.jld4") # corpus, deprel, postag
@date tst=load("acl11.tst.jld4")
@date trn=load("acl11.trn.jld4")
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
@save "$ftstr.jld" xtrn ytrn xdev ydev xtst ytst 
