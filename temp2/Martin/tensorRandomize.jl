########################################################################
#  CC-BY 2021 
#  Peter A Brooksbank
#  Martin Kassabov
#  James B. Wilson
#
#    Distributed under MIT License.
########################################################################


using LinearAlgebra
using SparseArrays
using Dates
using Random

#using Pkg
#Pkg.add("JLD")
#using JLD

include("tensorFunctions.jl")


# generate random transovection -- might also generate diagonal matrix
function randTrans(d)
    m = Matrix{Float32}(I,d,d)
    i = rand(1:d)
    j = rand(1:d)
	if i==j 
		# do no generate scalar matrices
		return m
	else
#        m[i,j] = rand(-1000:1000) * 0.001
        m[i,j] = randn(Float32)
    end 
    return m
end 

# generate random permutation matrix
function randomPermMarix(d)
    pi = randperm(d)
    m = Matrix{Float32}(I,d,d)
    for i = 1:d
		m[i,i] = 0
		m[i,pi[i]] = 1
    end
	return m
end 

# generate random matrix
function randomMatrix(d,rounds)
    mat = randomPermMarix(d)
    for i = 1:rounds
        mat *= randTrans(d)
    end
	return mat
end


# randomize a tensor
# rounds -- number of elemetary matrices applied to each side
function tensorRandomize(t,rounds )
    d1= size(t)[1]
	m1 = randomMatrix(d1, rounds)
    d2= size(t)[2]
	m2 = randomMatrix(d2, rounds)
    d3= size(t)[3]
	m3 = randomMatrix(d3, rounds)

    randt = actFirst( t, m1 )
    randt = actSecond( randt, m2 )
    randt = actThird( randt, m3 )
	return randt, [m1, m2, m3]
end

