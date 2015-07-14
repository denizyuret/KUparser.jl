using HDF5,JLD,KUparser,KUnet,Base.Test
require("mycat.jl")

if !isdefined(:corpus)
    info("Loading data")
    @date @load "yatbazp.tst.jld4"
end
@show ncpu = 10
@show nbatch = 10
@show ndeps = length(deprel)
@show ft = Flist.acl11eager
@show pt = ArcEager13
p0 = pt(1,ndeps)
s1 = corpus[1]

if !isdefined(:net)
    @date net = loadnet("testbparser4.net")
    # info("Let's train a reasonable model")
    # h0 = 20000
    # net = [Mmul(h0), Bias(), Relu(), 
    #        Mmul(p0.nmove), Bias(), Logp(), 
    #        LogpLoss()]
    # @date (p,x,y)=oparse(pt, corpus, ndeps, ft)
    # # @date train(net, x, y; loss=logploss)
    # @date train(net, x, y)
end

@show nbeam = 10
@date (q2,x2,y2)=bparse(pt, corpus[2:2], ndeps, ft, net, nbeam; xy=true)
@show map(size, (q2,x2,y2))
@show evalparse(q2, corpus[2:2])

if false 

info("Use gparser as reference for beam=1")
@date gxy=(gp,gx,gy)=gparse(pt, corpus, ndeps, ft, net; xy=true)
@show map(size, gxy)
@show evalparse(gp, corpus)

@show nbeam = 1

info("nbeam=1 Single sentence")
@date p1=bparse(pt, [s1], ndeps, ft, net, nbeam)
@test @show isequal(p1[1],gp[1])
@date (p2,x2,y2)=bparse(pt, [s1], ndeps, ft, net, nbeam; xy=true)
I2 = 1:size(x2,2)
@test @show isequal(p2[1],gp[1])
@test @show isequal(x2,gx[:,I2])
@test @show isequal(y2,gy[:,I2])

info("nbeam=1 Multiple sentences")
@date p3=bparse(pt, corpus, ndeps, ft, net, nbeam)
@test @show isequal(p3,gp)
@date pxy4=bparse(pt, corpus, ndeps, ft, net, nbeam; xy=true)
@show map(size, pxy4)
@show map(size, gxy)
@test @show isequal(pxy4[1],gxy[1])

info("nbeam=1 Batch processing")
@date p5=bparse(pt, corpus, ndeps, ft, net, nbeam, nbatch)
@test @show isequal(p5,gp)
@date pxy6=bparse(pt, corpus, ndeps, ft, net, nbeam, nbatch; xy=true)
@test @show pxyequal(pxy6, pxy4)

info("nbeam=1 Multi-cpu processing")
@date p7=bparse(pt, corpus, ndeps, ft, net, nbeam, nbatch, ncpu)
@test @show isequal(mycat(p7),gp)
@date pxy8=bparse(pt, corpus, ndeps, ft, net, nbeam, nbatch, ncpu; xy=true)
@test @show pxyequal(pxycat(pxy8), pxy4)

@show nbeam = 10

info("nbeam=10 Single sentence")
@date q1=bparse(pt, [s1], ndeps, ft, net, nbeam)
@show evalparse(q1, [s1])
@date (q2,x2,y2)=bparse(pt, [s1], ndeps, ft, net, nbeam; xy=true)
@show map(size, (q2,x2,y2))
@test @show isequal(q2,q1)

info("nbeam=10 Multiple sentences")
@date q3=bparse(pt, corpus, ndeps, ft, net, nbeam)
@show evalparse(q3, corpus)
@test @show isequal(q3[1],q1[1])
@date qxy4=(q4,x4,y4)=bparse(pt, corpus, ndeps, ft, net, nbeam; xy=true)
@show map(size, qxy4)
@test @show isequal(q4, q3)
I2 = 1:size(x2,2)
@test @show isequal(x4[:,I2], x2)
@test @show isequal(y4[:,I2], y2)

info("nbeam=10 Batch processing")
@date q5=bparse(pt, corpus, ndeps, ft, net, nbeam, nbatch)
@test @show isequal(q5,q3)
@date qxy6=bparse(pt, corpus, ndeps, ft, net, nbeam, nbatch; xy=true)
@show map(size, qxy6)
@test @show pxyequal(qxy6, qxy4)

info("nbeam=10 Multi-cpu processing")
@date q7=bparse(pt, corpus, ndeps, ft, net, nbeam, nbatch, ncpu)
@test @show isequal(mycat(q7),qxy6[1])
@date qxy8=bparse(pt, corpus, ndeps, ft, net, nbeam, nbatch, ncpu; xy=true)
@test @show pxyequal(pxycat(qxy8), qxy6)

end
