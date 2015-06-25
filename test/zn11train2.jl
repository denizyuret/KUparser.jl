using KUparser, KUnet, HDF5, JLD
KUnet.atype(Array)
@date @load "zn11oparse1.jld"
for y in (:ytrn, :ydev, :ytst); @eval $y = full($y); end
@show net = Layer[KPerceptron(size(ydev,1), KUnet.kpoly, [1,2])]
for i=1:100
    @date train(net, xtrn, ytrn; iters=100)
    @date println((i, size(net[1].s,2),
                   # accuracy(ytrn, predict(net, xtrn)), 
                   accuracy(ydev, predict(net, xdev)),
                   accuracy(ytst, predict(net, xtst)),
                   ))
end
:ok
