using KUnet, HDF5, JLD
isdefined(:xtrn) || (@date @load "zn11oparse1.jld")
nc = size(ydev,1)
g0 = 1.0
nbatch = 128
niters = 200
@show net = Layer[KPerceptron(nc, KUnet.kgauss, [g0])]
@date train(net, xdev, ydev; iters=niters, batch=nbatch)
@show (size(net[1].s), nbatch*niters)