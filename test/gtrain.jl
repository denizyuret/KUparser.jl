using HDF5, JLD, Dates, KUparser, KUnet

macro date(_x) :(println("$(now()) "*$(string(_x)));flush(STDOUT);@time $(esc(_x))) end
meminfo()=(@everywhere gc(); whos(); run(`nvidia-smi`); run(`ps auxww`|>`grep julia`); run(`free`))
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
    ncpu=12
    nbatch=2000
    feats=KUparser.Flist.fv021a

    net=KUnet.newnet(KUnet.relu, 1326, 20000, 3; learningRate=2f-2, adagrad=1f-8, dropout=7f-1)
    KUnet.setparam!(net[1]; dropout=2f-1)
    net[end].f=KUnet.logp
    @show net

    @date @load "conllWSJToken_wikipedia2MUNK-100.jld"
    @date initworkers(ncpu)

    @date p1=KUparser.oparse(trn, feats, ncpu)
    @show evalheads(p1,trn)
    @date p2=KUparser.oparse(dev, feats, ncpu)
    @show evalheads(p2,dev)
    @date p3=KUparser.oparse(tst, feats, ncpu)
    @show evalheads(p3,tst)

    for epoch=1:256
        @show epoch
        for q in p1
            @date KUnet.train(net, q[2], q[3]; batch=128, loss=KUnet.logploss)
        end
        p1=p2=p3=nothing
        meminfo()
        @date p1 = KUparser.gparse(trn, net, feats, nbatch, ncpu)
        @show e1 = evalheads(p1,trn)
        @date p2 = KUparser.gparse(dev, net, feats, nbatch, ncpu)
        @show e2 = evalheads(p2,dev)
        @date p3 = KUparser.gparse(tst, net, feats, nbatch, ncpu)
        @show e3 = evalheads(p3,tst)
        println("DATA:\t$epoch\t$e1\t$e2\t$e3"); flush(STDOUT)
    end
end

main()
