using HDF5,JLD,KUparser,Base.Test
eqparse(a,b)=(isequal(a[1],b[1]) && isequal(sortcols(vcat(a[2],a[3])), sortcols(vcat(b[2],b[3]))))
@date d = load("conll07.tst.jld4")
corpus = d["corpus"]
@show (ndeps,nbeam,nbatch,ncpu) = (length(d["deprel"]),10,20,5)
@show f1 = Flist.tacl13hybrid
s1 = corpus[1]
p1 = KUparser.Parser{:ArcHybrid}(wcnt(s1), ndeps)
net = KUnet.newnet(KUnet.relu, KUparser.flen(p1,s1,f1), 20000, p1.nmove; learningRate=1)
net[end].f = KUnet.logp
@date (p,x,y) = oparse(:ArcHybrid, corpus, f1, ndeps)
@date for i=1:10; KUnet.train(net, x, y; batch=128, loss=KUnet.logploss); end

@date g0 = gparse(:ArcHybrid, s1, net, f1, ndeps)
@date b0 = bparse(:ArcHybrid, s1, net, f1, ndeps, 1)
@test @show isequal(g0,b0)

@date g1 = gparse(:ArcHybrid, corpus, net, f1, ndeps, nbatch)
@show evalparse(g1[1], corpus)
@show map(size,g1)

@date g2 = gparse(:ArcHybrid, corpus, net, f1, ndeps)
@show isequal(g2,g1)
@test @show eqparse(g2,g1)
@date b2 = bparse(:ArcHybrid, corpus, net, f1, ndeps, 1)
@test @show isequal(b2,g1)

@date b3 = bparse(:ArcHybrid, corpus, net, f1, ndeps, 1, nbatch)
@test @show isequal(b3,g1)
@date b4 = bparse(:ArcHybrid, corpus, net, f1, ndeps, 1, nbatch, ncpu)
@show isequal(b4,g1)
@test @show eqparse(b4,g1)

@date b5 = bparse(:ArcHybrid, corpus, net, f1, ndeps, nbeam, nbatch)
@show evalparse(b5[1], corpus)
@show map(size, b5)
@show eqparse(b5, g1)
@date b6 = bparse(:ArcHybrid, corpus, net, f1, ndeps, nbeam, nbatch, ncpu)
@show isequal(b6,b5)
@test @show eqparse(b6,b5)
@date b7 = bparse(:ArcHybrid, corpus, net, f1, ndeps, nbeam)
@show isequal(b5,b7)
@show isequal(b6,b7)
@test @show eqparse(b5,b7)
@date b8 = KUparser.bparse1(:ArcHybrid, corpus, net, f1, ndeps, nbeam, ncpu)
@show isequal(b5,b8)
@show isequal(b6,b8)
@test @show isequal(b7,b8)
