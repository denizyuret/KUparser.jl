using HDF5,JLD,KUparser,Base.Test
@date d = load("conll07.tst.jld4")
@show ndeps = length(d["deprel"])
@date corpus = d["corpus"]
@date feats = Flist.acl11eager

# We try each arctype with rparse and oparse
# They will complain if movecosts or nmoves is buggy
for pt in (ArcEager13, ArcEagerR1, ArcHybrid13, ArcHybridR1)
    @show pt
    @date r = rparse(pt, corpus, ndeps)
    @show evalparse(r, corpus)
    @date (p,x,y) = oparse(pt, corpus, ndeps, feats)
    @show evalparse(p, corpus)
    p1 = pt(1, ndeps)
    @test @show size(x) == KUparser.xsize(p1, corpus, feats)
    @test @show size(y) == KUparser.ysize(p1, corpus)
end
