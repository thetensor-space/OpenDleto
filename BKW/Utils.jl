########################################################################
#  CC-BY 2021 
#  Peter A Brooksbank
#  Martin Kassabov
#  James B. Wilson
#
#    Distributed under MIT License.
########################################################################


###############################################################################
# Functions to randomize examples
###############################################################################


function matnorm( m)
    return map((x)->norm(x),m)
end

function matround( m, dig)
    return map((x)->round(norm(x),digits=dig),m)
end

#
# Permutes a tensor randomly.
#
function randomperm(ten)
    pi1 = randperm(size(ten)[1])
    pi2 = randperm(size(ten)[2])
    pi3 = randperm(size(ten)[3])

    ten2 = zeros(eltype(ten),size(ten))
    for i = 1:size(ten)[1]
        for j = 1:size(ten)[2]
            for k = 1:size(ten)[3]
                for m = 1:size(ten)[3]
                    ten2[i,j,k] = ten[pi1[i],pi2[j],pi3[k]]
                end
            end
        end
    end
    return ten2
end


#
# Returns a random I+a E_{ij} matrix (possibly with i=j but always nonsingular)
#
function randTrans(d)
    m = Matrix{Complex{Float16}}(I,d,d)
    i = rand(1:d)
    j = rand(1:d)
    v = complex(rand(Float16))
    if v != 0 
        m[i,j] = complex(rand(Float16))
    end 
    return m
end 

#
# Randomizes a tensor with sparse translations.
#
function sprandomize(t,rounds)
    ## 1 randomizedx
    d = size(t)[1]
#    mat = rand(Complex{Float16}, (d,d))
    mat = Matrix{Complex{Float16}}(I,d,d)
    for i = 1:rounds
        mat *= randTrans(size(t)[1])
    end
    t = actLeft( t, mat )

    ## 1 randomizedx
    d = size(t)[2]
#    mat = rand(Complex{Float16}, (d,d))
    mat = Matrix{Complex{Float16}}(I,d,d)
    for i = 1:rounds
        mat *= randTrans(size(t)[2])
    end
    t = actRight( t, mat )

    ## 3 randomized
    d = size(t)[3]
#    mat = rand(Complex{Float16}, (d,d))
    mat = Matrix{Complex{Float16}}(I,d,d)
    for i = 1:rounds 
        mat *= randTrans(size(t)[3])
    end
    t = actOut(t,mat)

    return randomperm(t)
end 

#
# Randomizes a tensor with dense translations.
#
function randomize(t)
    ## 1 randomized
    d = size(t)[1]
    mat = rand(Complex{Float16}, (d,d))
    t = actLeft( t, mat )

    ## 2 randomized
    d = size(t)[2]
    mat = rand(Complex{Float16}, (d,d))
    t = actRight( t, mat )

    ## 3 randomized
    d = size(t)[3]
    mat = rand(Complex{Float16}, (d,d))
    t = actOut(t,mat)

    return randomperm(t)
end 

#
# Return a tensor with sparse randome values
#
function sprandten(F, dims, density)
    return reshape( sprand(F,prod(dims), density), dims)
end

#
# Creates a random partition of n.
#
function randPartition(n)    
    top = 0
    ns = [top]
    while top < n 
        top = rand((top+1):n)
        ns = vcat(ns, [top])
    end
    return ns
end



#
# Make a 3-tensor that has blocks on its 12-face.
#
function makeSplit12( dims1, dims2, e )
    t = zeros(Complex{Float16}, dims1[length(dims1)], dims2[length(dims2)], e)
    for d = 1:(min(length(dims1),length(dims2))-1)
        for i = (dims1[d]+1):dims1[d+1]
            for j = (dims2[d]+1):dims2[d+1]
                for k = 1:e
                    t[i,j,k] = complex(rand(-3:3)) # rand(Float16))
                end
            end
        end
    end
    return t
end

