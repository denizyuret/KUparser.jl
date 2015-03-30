using HDF5,JLD,KUparser,Base.Test
@date @load "conll07.tst.jld4"
ft = Flist.acl11eager
pt = :ArcEager
ncpu = 10
ndeps = length(deprel)
s1 = corpus[1]
@date p1=oparse(pt, s1, ft, ndeps)
@date (p2,x2,y2)=oparse(pt, s1, ft, ndeps, true)
@test @show isequal(p2,p1)
@date p3=oparse(pt, corpus, ft, ndeps)
@test @show isequal(p3[1],p2)
@date (p4,x4,y4)=oparse(pt, corpus, ft, ndeps, true)
@test @show isequal(p4, p3)
I2 = 1:size(x2,2)
@test @show isequal(x4[:,I2], x2)
@test @show isequal(y4[:,I2], y2)
@date p5=oparse(pt, corpus, ft, ndeps, ncpu)
@test @show isequal(p5,p4)
@date (p6,x6,y6)=oparse(pt, corpus, ft, ndeps, ncpu, true)
@test @show isequal((p6,x6,y6), (p4,x4,y4))
