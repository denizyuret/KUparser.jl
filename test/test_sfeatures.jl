using HDF5,JLD,KUparser,KUnet,Base.Test

info("Load data")
@date @load "conll07.tst.jld4"
ncpu = 10
nbatch = 10
ndeps = length(deprel)
pt = ArcEager13
p0 = pt(1,ndeps)
s1 = corpus[1]

ft = Array[["s0w"], ["s0p"], ["s0w", "s0p"]]
@date (p1,x1,y1)=oparse(pt, s1, ndeps, ft)

SFarray = Array(Any, length(KUparser.SFhash))
for k in keys(KUparser.SFhash)
    v = KUparser.SFhash[k]
    SFarray[v] = k
end

for j=1:size(x1,2)
    println(SFarray[find(x1[:,j])])
end
