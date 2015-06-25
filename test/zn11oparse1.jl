using HDF5,JLD,KUparser,KUnet

info("Load data")
@date dev=load("acl11.dev.jld4")
@date tst=load("acl11.tst.jld4")
@date trn=load("acl11.trn.jld4")
@assert trn["deprel"] == dev["deprel"] == tst["deprel"]
@assert trn["postag"] == dev["postag"] == tst["postag"]
nd = length(dev["deprel"])
pt = ArcEager13
ft = Flist.zn11single
@date (pdev,xdev,ydev) = oparse(pt, dev["corpus"], nd, ft)
@date (ptst,xtst,ytst) = oparse(pt, tst["corpus"], nd, ft)
@date (ptrn,xtrn,ytrn) = oparse(pt, trn["corpus"], nd, ft)
nrows = max(maximum(xtrn.rowval), maximum(xtst.rowval), maximum(xdev.rowval))
xtrn.m = xtst.m = xdev.m = nrows
ytrn = sparse(ytrn); ydev = sparse(ydev); ytst = sparse(ytst)
@save "zn11oparse1.jld" xtrn ytrn xdev ydev xtst ytst 
# @save "zn11oparse1_hash.jld" KUparser.SFhash

# @date (pdev,xdev,ydev) = oparse(pt, dev["corpus"][1:100], nd, ft)
# @save "foo.zn11orig.jld" xdev ydev

:ok

# p0 = pt(1,ndeps)
# s1 = corpus[1]

# ft = Array[["s0w"], ["s0p"], ["s0w", "s0p"]]
# @date (p1,x1,y1)=oparse(pt, s1, ndeps, ft)

# SFarray = Array(Any, length(KUparser.SFhash))
# for k in keys(KUparser.SFhash)
#     v = KUparser.SFhash[k]
#     SFarray[v] = k
# end

# for j=1:size(x1,2)
#     println(SFarray[find(x1[:,j])])
# end
