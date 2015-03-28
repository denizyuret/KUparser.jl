# Testing for ncpu x pbatch x nbeam that fits in GPU memory
using HDF5,JLD,KUparser,Base.Test
eqparse(a,b)=(isequal(a[1],b[1]) && isequal(sortcols(vcat(a[2],a[3])), sortcols(vcat(b[2],b[3]))))
@date d = load("acl11.trn.jld4")
corpus = d["corpus"]
@show (ndeps,nbeam,nbatch,ncpu) = (length(d["deprel"]),10,64,12)
@show f1 = Flist.acl11eager
s1 = corpus[1]
p1 = KUparser.Parser{:ArcEager}(wcnt(s1), ndeps)
@show net = KUnet.newnet(KUnet.relu, KUparser.flen(p1,s1,f1), 20000, p1.nmove; learningRate=1)
net[end].f = KUnet.logp
@date (p,x,y) = oparse(:ArcEager, corpus, f1, ndeps, ncpu)
@show map(size, (x,y))
@date KUnet.train(net, x, y; batch=128, loss=KUnet.logploss)
@date b0 = bparse(:ArcEager, corpus, net, f1, ndeps, nbeam, nbatch, ncpu)
