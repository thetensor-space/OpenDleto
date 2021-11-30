########################################################################
#  CC-BY 2021 
#  Peter A Brooksbank
#  Martin Kassabov
#  James B. Wilson
#
#    Distributed under MIT License.
########################################################################


using Random


include("TensorFunctions.jl")


# random number generator with values -1 to 1
function randomNumber(;spread=1,type="uniform")
	if type=="uniform"
		return spread*(2*rand() - 1)
	end
	if type=="normal"
		return spread*randn()
	end
	#safety
	return 0.5   
end 



# generate random transovection -- might also generate diagonal matrix
function randTrans(d;type="normal")
    m = Matrix{Float32}(I,d,d)
    i = rand(1:d)
    j = rand(1:d)
	if i==j 
		# do no generate scalar matrices
		return m
	else
		m[i,j] = randomNumber(;type)
    end 
	return m  
end 

# generate random permutation matrix
function randomPermMarix(d)
    pi = randperm(d)
    m = zeros(d,d)
    for i = 1:d
#		m[i,i] = 0
		m[i,pi[i]] = 1
    end
	return m
end 

# generate random matrix
function randomMatrix(d,rounds;type="normal")
    mat = randomPermMarix(d)
    for i = 1:rounds
        mat *= randTrans(d;type)
    end
	return mat
end


# randomize a tensor
# rounds -- number of elemetary matrices applied to each side
function tensorRandomize(t,rounds;type="normal")
    d1= size(t)[1]
	m1 = randomMatrix(d1, rounds[1]; type)
    d2= size(t)[2]
	m2 = randomMatrix(d2, rounds[2]; type)
    d3= size(t)[3]
	m3 = randomMatrix(d3, rounds[3]; type)

	return actAll(t,[m1, m2, m3]), [m1, m2, m3]
end

# add noise to a tensor
function tensorAddNoise(t,rounds; type="normal", relsize=0.001)
	ten = copy(t)
	norm = tensorMaxEntry(t)
	for i = 1: rounds
		p = rand(1:size(ten)[1])
		q = rand(1:size(ten)[2])
		r = rand(1:size(ten)[2])
		ten[p,q,r] += norm*relsize*randomNumber(;type)
	end
	return ten
end

nothing;