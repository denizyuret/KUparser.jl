# This restarts workers, need this until they fix memory leaks.
# Unfortunately it doesn't work from within the module.
function resetworkers(ncpu)
    nprocs() > 1 && (rmprocs(workers());sleep(2))
    addprocs(ncpu)
    require("CUDArt")
    @everywhere CUDArt.device((myid()-1) % CUDArt.devcount())
    require("CUBLAS")
    require("KUnet")
    require("KUparser")
end # resetworkers
