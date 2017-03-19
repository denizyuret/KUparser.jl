# This restarts workers, need this until they fix memory leaks.
# Unfortunately it doesn't work from within the module.
function resetworkers(ncpu)
    rmworkers()
    addprocs(ncpu)
    require("CUDArt")
    @everywhere CUDArt.device((myid()-1) % CUDArt.devcount())
    require("CUBLAS")
    require("Knet")
    require("KUparser")
end # resetworkers

# Really remove all workers
# There is also Base.terminate_all_workers() we can use here...
function rmworkers()
    nprocs() > 1 && rmprocs(workers())
    while nworkers() > 1
        sleep(1)
    end
end

function restartmachines()
    machines = ASCIIString[]
    for i in keys(Base.map_pid_wrkr)
        i == 1 && continue
        w = Base.map_pid_wrkr[i]
        push!(machines, w.host)
    end
    Base.terminate_all_workers()
    addprocs(machines)
end

# This does not distribute gpus equally:
# @date @everywhere using CUDArt
# @date @everywhere CUDArt.device((myid()-1) % CUDArt.devcount())

function restartcuda()
    @everywhere require("CUDArt")
    d = Dict()
    for i in keys(Base.map_pid_wrkr)
        i == 1 && continue
        a = Base.map_pid_wrkr[i].bind_addr
        n = get!(d, a, 0)
        # @show (a,n)
        @fetchfrom i CUDArt.device(n % CUDArt.devcount())
        d[a] = n+1
    end
end

function restartmachines(host::ASCIIString)
    wid = 0
    for i in keys(Base.map_pid_wrkr)
        i == 1 && continue
        w = Base.map_pid_wrkr[i]
        w.host == host && (wid = i; break)
    end
    if wid == 0 
        warn("Cannot find worker on $host")
        return nothing
    end
    ret = rmprocs(wid; waitfor=0.5)
    if ret != :ok
        warn("Forcibly interrupting busy worker $i")
        # Might be computation bound, interrupt them and try again
        interrupt(wid)
        ret = rmprocs(wid; waitfor=0.5)
        if ret != :ok
            warn("Unable to terminate worker $i")
            return nothing
        end
    end
    @assert ret == :ok
    return addprocs([host])[1]
end

function restartmachines(machines::AbstractVector)
    map(restartmachines, machines)
end
