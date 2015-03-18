using HDF5, JLD, Dates, KUparser, KUnet

macro date(_x) :(println("$(now()) "*$(string(_x)));flush(STDOUT);@time $(esc(_x))) end
macro meminfo() :(gc(); run(`nvidia-smi`); run(`ps auxww`|>`grep julia`); run(`free`)) end
evalheads(p,c)=mean(vcat(vcat(map(q->q[1],p)...)...) .== vcat(map(s->s.head,c)...))

function initworkers(ncpu)
    (nworkers() < ncpu) && (addprocs(ncpu - nprocs() + 1))
    require("CUDArt")
    @everywhere CUDArt.device((myid()-1) % CUDArt.devcount())
    require("CUBLAS")
    require("KUnet")
    require("KUparser")
end

function main()
    @date @load "conllWSJToken_wikipedia2MUNK-100.jld"
    ncpu=12
    nbatch=200
    nbeam=10
    nsent=length(trn)
    nblock=int(ceil(nsent/nbeam))
    feats=KUparser.Flist.fv021a

    @date initworkers(ncpu)
    @date p1=KUparser.oparse(trn, feats, ncpu)
    @show evalheads(p1,trn)
    @date p2=KUparser.oparse(dev, feats, ncpu)
    @show evalheads(p2,dev); p2=nothing
    @date p3=KUparser.oparse(tst, feats, ncpu)
    @show evalheads(p3,tst); p3=nothing
    @date rmprocs(workers())
    sleep(5)
    @meminfo

    net=KUnet.newnet(KUnet.relu, 1326, 20000, 3; learningRate=2f-2, adagrad=1f-8, dropout=7f-1)
    KUnet.setparam!(net[1]; dropout=2f-1)
    net[end].f=KUnet.logp
    @show net
    for q in p1
        @date KUnet.train(net, q[2], q[3]; batch=128, loss=KUnet.logploss)
    end
    p1=nothing

    for epoch=1:256
        @show epoch
        @meminfo
        e1=0.0; nb=0
        for s1=1:nblock:nsent
            s2=min(nsent,s1+nblock-1)
            s12=sub(trn,s1:s2)
            @date initworkers(ncpu)
            @date p1 = KUparser.bparse(s12, net, feats, nbeam, nbatch, ncpu)
            @date rmprocs(workers())
            nb += 1 
            @show e1 = (1-1/nb)*e1 + (1/nb)*evalheads(p1,s12)
            for q in p1
                @date KUnet.train(net, q[2], q[3]; batch=128, loss=KUnet.logploss)
            end
            p1=nothing
            @meminfo
        end
        @date initworkers(ncpu)
        @date p2 = KUparser.bparse(dev, net, feats, nbeam, nbatch, ncpu)
        @show e2 = evalheads(p2,dev); p2=nothing
        @date p3 = KUparser.bparse(tst, net, feats, nbeam, nbatch, ncpu)
        @show e3 = evalheads(p3,tst); p3=nothing
        @date rmprocs(workers())
        sleep(5)
        @meminfo
        println("DATA:\t$epoch\t$e1\t$e2\t$e3"); flush(STDOUT)
    end
end

main()
