using HDF5,JLD,KUparser,Base.Test
include("pxyequal.jl")

# All interfaces give the same result
info("Load data")
@date @load "conll07.tst.jld4"
ft = Flist.acl11eager
pt = ArcEager13
ncpu = 10
ndeps = length(deprel)
s1 = corpus[1]

info("Single sentence")
@date p1=oparse(pt, [s1], ndeps)
info("with features")
@date pxy2=(p2,x2,y2)=oparse(pt, [s1], ndeps, ft)
@test @show isequal(p2,p1)

info("Multiple sentences")
@date p3=oparse(pt, corpus, ndeps)
@test @show isequal(p3[1],p2[1])
info("with features")
@date pxy4=(p4,x4,y4)=oparse(pt, corpus, ndeps, ft)
@test @show isequal(p4, p3)
I2 = 1:size(x2[1],2)
@test @show isequal(x4[1][:,I2], x2[1])
@test @show isequal(y4[1][:,I2], y2[1])

info("Multi-cpu processing")
@date p5=oparse(pt, corpus, ndeps; usepmap=true)
@test @show isequal(p5,p4)
info("with features")
@date pxy6=(p6,x6,y6)=oparse(pt, corpus, ndeps, ft; usepmap=true)
@test @show pxyequal(pxy6,pxy4)
