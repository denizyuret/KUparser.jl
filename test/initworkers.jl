(nworkers() < ncpu) && (addprocs(ncpu - nprocs() + 1))
require("CUDArt")
@everywhere CUDArt.device((myid()-1) % CUDArt.devcount())
require("CUBLAS")
require("KUnet")
require("KUparser")

