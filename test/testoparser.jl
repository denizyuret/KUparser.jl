using HDF5,JLD,KUparser,Base.Test

# All interfaces give the same result
info("Load data")
@date @load "conll07.tst.jld4"
ft = Flist.acl11eager
pt = ArcEager13
ncpu = 10
ndeps = length(deprel)
s1 = corpus[1]

info("Single sentence")
@date p1=oparse(pt, s1, ndeps)
info("with features")
@date (p2,x2,y2)=oparse(pt, s1, ndeps, ft)
@test @show isequal(p2,p1)

info("Multiple sentences")
@date p3=oparse(pt, corpus, ndeps)
@test @show isequal(p3[1],p2)
info("with features")
@date (p4,x4,y4)=oparse(pt, corpus, ndeps, ft)
@test @show isequal(p4, p3)
I2 = 1:size(x2,2)
@test @show isequal(x4[:,I2], x2)
@test @show isequal(y4[:,I2], y2)

info("Multi-cpu processing")
@date p5=oparse(pt, corpus, ndeps, ncpu)
@test @show isequal(p5,p4)
info("with features")
@date (p6,x6,y6)=oparse(pt, corpus, ndeps, ncpu, ft)
@test @show isequal((p6,x6,y6), (p4,x4,y4))
