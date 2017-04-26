# Finding memory usage:

_getbytes(x::DataType,d)=[sizeof(Int),0]
_getbytes(x::NTuple,d)=sum(map(y->_getbytes(y,d), x))
_getbytes(x::KnetArray,d)=(haskey(d,x) ? [0,0] : (d[x]=1; [0, length(x) * sizeof(eltype(x))])) # this does not count the KnetArray struct, do KnetPtr instead?
_getbytes(x::Symbol,d)=[sizeof(Int),0]

function _getbytes(x::DenseArray,d) 
    haskey(d,x) && return [0,0]; d[x]=1
    total = [sizeof(x),0]
    if !isbits(eltype(x))
        for i = 1:length(x)
            isassigned(x,i) || continue
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
    total = [sizeof(x),0]
    isbits(x) && return total
    haskey(d,x) && return 0; d[x]=1
    fieldNames = fieldnames(x)
    if fieldNames != ()
        for fieldName in fieldNames
            isdefined(x, fieldName) || continue
            f = getfield(x,fieldName)
            fieldBytes = _getbytes(f,d)
            # fieldSize = (isa(f, Union{AbstractArray,KnetArray}) ? (eltype(f),size(f)) : ())
            # @show (typeof(x), fieldName, fieldSize, fieldBytes)
            total += fieldBytes
        end
    end
    return total
end

getbytes(x)=tuple(_getbytes(x, ObjectIdDict())...)
