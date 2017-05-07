conflat(wij, id, maxlen) = vcat(map(x->vcat(x, fill!(Array(typeof(x[1]), maxlen - length(x)), id)), wij)...)

function batch4conv(wij, maxlen, sow, eow, pad=eow)
    wij2 = map(x->vcat(sow, x, eow), wij)
    convdata = conflat(wij2, pad, maxlen)

    convmasks = map(x->fill!(similar(x), 1), wij2)
    convmasks = conflat(convmasks, 0, maxlen)
    return (convdata, convmasks)
end

function convembed(model, convdata, mask, lw, d, pwin)
    batchsize = Int(length(convdata) / lw)
    mcembed, mfbank, mbias = model[1], model[2], model[3]

    mask2 = (typeof(mcembed) <: KnetArray ? convert(KnetArray{Float32}, mask) : mask) 
    
    to_embed = []
    for i=1:lw:length(convdata)
        win = convdata[i:(i+lw-1)]
        x = mcembed[win, :]
        x1 = x .* mask2[i:(i+lw-1)]
        push!(to_embed, x1)
    end
    emw = hcat(to_embed...)
    c_k = reshape(emw, (lw, d, 1, batchsize))
    y_k = pool(tanh(conv4(mfbank, c_k) .+ mbias); window=pwin)
    return mat(y_k)

end

#=
wmodel is an array of Deniz hoca's here is the conversion style, all the expressions below are true
I am not transposing the cembed because at the end of convolution we have a matrix of embedsize x batchsize matrix
wmodel[1] == model[:cembed]
wmodel[2] == model[:conv][1] ; wmodel[3] == model[:conv][2]
wmodel[4] == model[:forw][1]'; wmodel[5] == model[:forw][2]'
wmodel[6] == model[:back][1]'; wmodel[7] == model[:back][2]'
wmodel[8] == model[:soft][1]'; wmodel[9] == model[:soft][2]'

=#

##### Dead code
#vcat(map(x->vcat(x, fill!(Array(typeof(x[1]), maxlen - length(x)), pad)), wij)...)
