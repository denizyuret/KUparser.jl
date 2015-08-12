# Test the calling interface for parsers

using Compat
using HDF5,JLD,KUparser,KUnet,Base.Test
infoln(x)=(println();info(x))
include("pxyequal.jl")

infoln("Loading data")
@date @load "conll07.tst.jld4"
@show nbatch = 4
@show nbeam = 4
@show ndeps = length(deprel)
@show ft = Flist.acl11eager
@show pt = ArcEager13
@show wt = wtype(corpus)
@show pxytuple = @compat Tuple{Vector{pt}, Vector{Matrix{wt}}, Vector{Matrix{wt}}}
p0 = pt(1,ndeps)
s1 = corpus[1]

infoln("Let's train a reasonable model")
h0 = 20000
net = [Mmul(h0), Bias(), Relu(), 
       Mmul(p0.nmove), Bias(), Logp(), 
       LogpLoss()]
@date (p,x,y)=oparse(pt, corpus, ndeps, ft)
@date train(net, hcat(x...), hcat(y...))
@date c = corpus[1:16]

infoln("Testing oparser:")
@date o0 = oparse(pt, c, ndeps); @test isa(o0, Vector{pt})
@date o1 = oparse(pt, c, ndeps; usepmap=true); @test isequal(o1,o0)
@date o2 = oparse(pt, c, ndeps, ft); @test isa(o2, pxytuple); @test isequal(o2[1],o0)
@date o3 = oparse(pt, c, ndeps, ft; usepmap=true); @test pxyequal(o3,o2)

infoln("Testing gparser:")
@date g0 = gparse(pt, c, ndeps, ft, net); @test isa(g0, Vector{pt})
@date g1 = gparse(pt, c, ndeps, ft, net; nbatch=nbatch); @test isequal(g1,g0)
@date g2 = gparse(pt, c, ndeps, ft, net; usepmap=true); @test isequal(g2,g0)
@date g3 = gparse(pt, c, ndeps, ft, net; usepmap=true,nbatch=nbatch); @test isequal(g3,g0)
@date g4 = gparse(pt, c, ndeps, ft, net; returnxy=true); @test isa(g4, pxytuple); @test isequal(g4[1],g0)
@date g5 = gparse(pt, c, ndeps, ft, net; returnxy=true,nbatch=nbatch); @test pxyequal(g5,g4)
@date g6 = gparse(pt, c, ndeps, ft, net; returnxy=true,usepmap=true); @test pxyequal(g6,g4)
@date g7 = gparse(pt, c, ndeps, ft, net; returnxy=true,usepmap=true,nbatch=nbatch); @test pxyequal(g7,g4)

infoln("Testing bparser:")
@date bx = bparse(pt, c, ndeps, ft, net, 1); @test isequal(bx, g0)
@date b0 = bparse(pt, c, ndeps, ft, net, nbeam); @test isa(b0, Vector{pt})
@date b1 = bparse(pt, c, ndeps, ft, net, nbeam; nbatch=nbatch); @test isequal(b1,b0)
@date b2 = bparse(pt, c, ndeps, ft, net, nbeam; usepmap=true); @test isequal(b2,b0)
@date b3 = bparse(pt, c, ndeps, ft, net, nbeam; usepmap=true,nbatch=nbatch); @test isequal(b3,b0)
@date b4 = bparse(pt, c, ndeps, ft, net, nbeam; returnxy=true); @test isa(b4, pxytuple)
@date b5 = bparse(pt, c, ndeps, ft, net, nbeam; returnxy=true,nbatch=nbatch); @test pxyequal(b5,b4)
@date b6 = bparse(pt, c, ndeps, ft, net, nbeam; returnxy=true,usepmap=true); @test pxyequal(b6,b4)
@date b7 = bparse(pt, c, ndeps, ft, net, nbeam; returnxy=true,usepmap=true,nbatch=nbatch); @test pxyequal(b7,b4)

:ok