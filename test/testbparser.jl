using HDF5,JLD,KUparser,KUnet,Base.Test
require("pxyequal.jl")
infoln(x)=(println();info(x))

infoln("Loading data")
@date @load "conll07.tst.jld4"
@show nbatch = 10
@show ndeps = length(deprel)
@show ft = Flist.acl11eager
@show pt = ArcEager13
p0 = pt(1,ndeps)
s1 = corpus[1]

infoln("Let's train a reasonable model")
h0 = 20000
net = [Mmul(h0), Bias(), Relu(), 
       Mmul(p0.nmove), Bias(), Logp(), 
       LogpLoss()]
@date (p,x,y)=oparse(pt, corpus, ndeps, ft)
@date train(net, hcat(x...), hcat(y...))

infoln("0. Use gparser as reference for beam=1")
@date pxy0=(p0,x0,y0)=gparse(pt, corpus, ndeps, ft, net; returnxy=true)
@show map(size, pxy0)
@show evalparse(p0, corpus)

println()
@show nbeam = 1

infoln("1. nbeam=1 Single sentence")
@date p1=bparse(pt, [s1], ndeps, ft, net, nbeam)
@test @show isequal(p1[1],p0[1])

infoln("2. nbeam=1 Single sentence, xy")
@date pxy2=(p2,x2,y2)=bparse(pt, [s1], ndeps, ft, net, nbeam; returnxy=true)
@show map(size, pxy2)
@show I2 = 1:size(x2[1],2)
infoln("these may not be equal because of early stop")
@show isequal(p2[1],p0[1])
@show isequal(x2[1],x0[:,I2])
@show isequal(y2[1],y0[:,I2])

infoln("3. nbeam=1 Multiple sentences")
@date p3=bparse(pt, corpus, ndeps, ft, net, nbeam)
@test @show isequal(p3,p0)

infoln("4. nbeam=1 Multiple sentences, xy")
@date pxy4=(p4,x4,y4)=bparse(pt, corpus, ndeps, ft, net, nbeam; returnxy=true)
@show map(size, pxy4)
@show isequal(p4,p0)
@test @show isequal(p4[1],p2[1])
@test @show isequal(x4[1][:,I2],x2[1])
@test @show isequal(y4[1][:,I2],y2[1])

infoln("5. nbeam=1 Batch processing")
@date p5=bparse(pt, corpus, ndeps, ft, net, nbeam; nbatch=nbatch)
@test @show isequal(p5,p0)

infoln("6. nbeam=1 Batch processing, xy")
@date pxy6=(p6,x6,y6)=bparse(pt, corpus, ndeps, ft, net, nbeam; nbatch=nbatch, returnxy=true)
@show map(size,pxy6)
@test @show pxyequal(pxy6, pxy4)

if nworkers() > 1
    infoln("7. nbeam=1 Multi-cpu processing")
    @date p7=bparse(pt, corpus, ndeps, ft, net, nbeam; nbatch=nbatch, usepmap=true)
    @test @show isequal(p7,p0)
    infoln("8. nbeam=1 Multi-cpu processing, xy")
    @date pxy8=bparse(pt, corpus, ndeps, ft, net, nbeam; nbatch=nbatch, usepmap=true, returnxy=true)
    @test @show pxyequal(pxy8, pxy4)
else
    infoln("Use 'julia -p n' to test multi-cpu.")
end

println()
@show nbeam = 10

infoln("1. nbeam=10 Single sentence")
@date q1=bparse(pt, [s1], ndeps, ft, net, nbeam)
@show evalparse(q1, [s1])

infoln("2. nbeam=10 Single sentence, xy")
@date qxy2=(q2,x2,y2)=bparse(pt, [s1], ndeps, ft, net, nbeam; returnxy=true)
@show map(size, (q2,x2,y2))
@show I2 = 1:size(x2[1],2)
@show isequal(q2,q1) # may not be equal due to early stop

infoln("3. nbeam=10 Multiple sentences")
@date q3=bparse(pt, corpus, ndeps, ft, net, nbeam)
@show evalparse(q3, corpus)
@test @show isequal(q3[1],q1[1])

infoln("4. nbeam=10 Multiple sentences, xy")
@date qxy4=(q4,x4,y4)=bparse(pt, corpus, ndeps, ft, net, nbeam; returnxy=true)
@show map(size, qxy4)
@show isequal(q4, q3)           # may not equal, earlystop
@test @show isequal(q4[1], q2[1])
@test @show isequal(x4[1][:,I2], x2[1])
@test @show isequal(y4[1][:,I2], y2[1])

infoln("5. nbeam=10 Batch processing")
@date q5=bparse(pt, corpus, ndeps, ft, net, nbeam; nbatch=nbatch)
@test @show isequal(q5,q3)

infoln("6. nbeam=10 Batch processing, xy")
@date qxy6=(q6,x6,y6)=bparse(pt, corpus, ndeps, ft, net, nbeam; nbatch=nbatch, returnxy=true)
@show map(size, qxy6)
@test @show pxyequal(qxy6, qxy4)

if nworkers() > 1
    infoln("7. nbeam=10 Multi-cpu processing")
    @date q7=bparse(pt, corpus, ndeps, ft, net, nbeam; nbatch=nbatch, usepmap=true)
    @test @show isequal(q7, q3)
    infoln("8. nbeam=10 Multi-cpu processing, xy")
    @date qxy8=bparse(pt, corpus, ndeps, ft, net, nbeam; nbatch=nbatch, usepmap=true, returnxy=true)
    @test @show pxyequal(qxy8, qxy6)
else
    infoln("Use 'julia -p n' to test multi-cpu.")
end
