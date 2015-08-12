using HDF5,JLD,KUparser,KUnet,Base.Test
include("pxyequal.jl")

info("Load data")
@date @load "conll07.tst.jld4"
ncpu = 10
nbatch = 10
ndeps = length(deprel)
ft = Flist.acl11eager
pt = ArcEager13
p0 = pt(1,ndeps)
s1 = corpus[1]

info("Let's train a reasonable model")
# net = newnet(relu, flen(p0,s1,ft), 20000, p0.nmove; learningRate=1)
# net[end].f = logp
h0 = 20000
net = [Mmul(h0), Bias(), Relu(), 
       Mmul(p0.nmove), Bias(), Logp(), 
       LogpLoss()]
@date (p,x,y)=oparse(pt, corpus, ndeps, ft)
@date train(net, hcat(x...), hcat(y...))

info("Single sentence")
@date p1=gparse(pt, [s1], ndeps, ft, net)
@show evalparse(p1, [s1])
@date (p2,x2,y2)=gparse(pt, [s1], ndeps, ft, net; returnxy=true)
@show map(size, (p2,x2,y2))
@show I2 = 1:size(x2[1],2)
@test @show isequal(p2,p1)

info("Multiple sentences")
@date p3=gparse(pt, corpus, ndeps, ft, net)
@show evalparse(p3, corpus)
@test @show isequal(p3[1],p1[1])
@date pxy4=(p4,x4,y4)=gparse(pt, corpus, ndeps, ft, net; returnxy=true)
@show map(size, pxy4)
@test @show isequal(p4, p3)
@test @show isequal(x4[1][:,I2], x2[1])
@test @show isequal(y4[1][:,I2], y2[1])

info("Batch processing")
@date p5=gparse(pt, corpus, ndeps, ft, net; nbatch=nbatch)
@test @show isequal(p5,p3)
@date pxy6=gparse(pt, corpus, ndeps, ft, net; nbatch=nbatch, returnxy=true)
@show map(size, pxy6)
@test @show pxyequal(pxy6, pxy4)

info("Multi-cpu processing")
@date p7=gparse(pt, corpus, ndeps, ft, net; nbatch=nbatch, usepmap=true)
@test @show isequal(p7,pxy6[1])
@date pxy8=gparse(pt, corpus, ndeps, ft, net; nbatch=nbatch, returnxy=true, usepmap=true)
@test @show pxyequal(pxy8, pxy6)
