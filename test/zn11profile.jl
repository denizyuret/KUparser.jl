using KUparser, KUnet, HDF5, JLD
@date @load "zn11oparse.jld"
@show KUnet.atype(Array)
@show niter = 0
@show net = Layer[Perceptron(size(ydev,1))]
@date train(net, xdev, ydev; iters=niter)
@show net = Layer[Perceptron(size(ydev,1))]
@date train(net, xdev, ydev; iters=niter)
@show net = Layer[Perceptron(size(ydev,1))]
@date @profile train(net, xdev, ydev; iters=niter)
Profile.print()

# for i=1:100
#     @date train(net, xtst, ytst)
#     @date println((i, accuracy(ytst, predict(net, xtst)), 
#                    accuracy(ydev, predict(net, xdev))))
# end
# :ok
