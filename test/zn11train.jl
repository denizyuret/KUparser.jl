using KUparser, KUnet, HDF5, JLD
KUnet.atype(Array)
@date @load "zn11oparse.jld"
for y in (:ytrn, :ydev, :ytst); @eval $y = full($y); end
@show net = Layer[Perceptron(size(ydev,1))]
for i=1:100
    @date train(net, xtrn, ytrn; shuffle=true)
    @date println((i, 
                   accuracy(ytrn, predict(net, xtrn)), 
                   accuracy(ydev, predict(net, xdev)),
                   accuracy(ytst, predict(net, xtst)),
                   ))
end
:ok
