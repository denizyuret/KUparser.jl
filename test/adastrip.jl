using KUnet
net = loadnet(ARGS[1])
display(net); println();
for l in net
    for n in names(l)
        if isdefined(l,n) && isa(l.(n), KUparam)
            l.(n) = KUparam(l.(n).arr)
        end
    end
end
display(net); println();
savenet(ARGS[1]*".adastrip", net)
