function pxycat(pxy)
    # concat parsing results from multiple cpus
    # input [(p1,x1,y1), (p2,x2,y2), ...]
    # output (p, x, y)
    (mycat(map(z->z[1], pxy)),
     mycat(map(z->z[2], pxy)),
     mycat(map(z->z[3], pxy)))
end

function mycat(a::Array)
    # concat elements of a along their last dimension
    mx = size(a[1])[1:end-1]
    nx = 0
    for x in a
        @assert size(x)[1:end-1] == mx
        nx += size(x)[end]
    end
    b = similar(a[1], tuple(mx..., nx))
    nb = 0
    for x in a
        copy!(b, nb+1, x, 1, length(x))
        nb += length(x)
    end
    return b
end
