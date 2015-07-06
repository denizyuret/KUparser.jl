using CUDArt

_getbytes(x::DataType,d) = sizeof(x)

_getbytes(x::NTuple,d)=(length(x) * sizeof(eltype(x)))

_getbytes(x::AbstractCudaArray,d)=(haskey(d,x) ? 0 : (d[x]=1; length(x) * sizeof(eltype(x))))

function _getbytes(x::DenseArray,d) 
    haskey(d,x) && return 0; d[x]=1
    total = sizeof(x)
    if !isbits(eltype(x))
        for i = 1:length(x)
            isize = _getbytes(x[i],d)
            # @show (typeof(x), i, isize)
            total += isize
        end
    end
    # @show (typeof(x), total)
    # @show (eltype(x), size(x), total)
    return total
end

function _getbytes(x,d)
    total = sizeof(x)
    isbits(x) && return total
    haskey(d,x) && return 0; d[x]=1
    fieldNames = typeof(x).names
    if fieldNames != ()
        for fieldName in fieldNames
            isdefined(x, fieldName) || continue
            f = x.(fieldName)
            fieldBytes = _getbytes(f,d)
            fieldSize = (isa(f, Union(AbstractArray,CudaArray)) ? (eltype(f),size(f)) : ())
            # @show (typeof(x), fieldName, fieldSize, fieldBytes)
            total += fieldBytes
        end
    end
    return total
end

getbytes(x)=_getbytes(x, ObjectIdDict())