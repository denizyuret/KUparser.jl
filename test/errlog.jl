# From: https://thenewphalls.wordpress.com/tag/julia/

(errRead, errWrite) = redirect_stderr()
(outRead, outWrite) = redirect_stdout()
 
function flushlog()
    global errRead
    global errWrite
    global outRead
    global outWrite
    dir = "/home/nlg-05/dy_052/kuparser/dev/test"

    close(errWrite)
    err = readavailable(errRead)
    close(errRead)
    errfile = open("$(dir)/$(myid()).err", "a")
    write(errfile, err)
    close(errfile)

    close(outWrite)
    out = readavailable(outRead)
    close(outRead)
    outfile = open("$(dir)/$(myid()).out", "a")
    write(outfile, out)
    close(outfile)
end

atexit(flushlog)
