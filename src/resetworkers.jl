# This restarts workers, need this until they fix memory leaks.
# Unfortunately it doesn't work from within the module.
function resetworkers(ncpu)
    rmworkers()
    addprocs(ncpu)
    require("CUDArt")
    @everywhere CUDArt.device((myid()-1) % CUDArt.devcount())
    require("CUBLAS")
    require("KUnet")
    require("KUparser")
end # resetworkers

# Really remove all workers
function rmworkers()
    nprocs() > 1 && rmprocs(workers())
    while nworkers() > 1
        sleep(1)
    end
end
