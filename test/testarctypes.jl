using HDF5,JLD,KUparser
@date d = load("conll07.tst.jld4")
@show ndeps = length(d["deprel"])
@date corpus = d["corpus"]
@date s1 = corpus[1]
for pt in (ArcEager13, ArcEagerR1, ArcHybrid13, ArcHybridR1)
    @show pt
    @date p1 = pt(wcnt(s1), ndeps)
    @date r0 = rparse(pt, corpus, ndeps)
    @show evalparse(r0, corpus)
    @date f1 = Flist.tacl13hybrid
    @date r1 = oparse(pt, corpus, f1, ndeps)
    @show evalparse(r1, corpus)
    @date f2 = Flist.acl11eager
    @date r2 = oparse(pt, corpus, f2, ndeps)
    @show evalparse(r2, corpus)
    # @date net = KUnet.newnet(KUnet.relu, KUparser.flen(p1,s1,f1), 20000, p1.nmove)
    # @date r3 = gparse(pt, corpus, net, f1, ndeps)
    # @show evalparse(r3, corpus)
    # @date r4 = gparse(pt, corpus, net, f1, ndeps, 9999)
    # @show evalparse(r4, corpus)
end
