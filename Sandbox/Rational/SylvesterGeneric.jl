
#
#    A demonstration of a Sylvester solver.
# 
#    Copyright 2021 Amaury V. Miniño and James B. Wilson.
#    Distributed under MIT License.
#

using LinearAlgebra

# In Matrix notation
#  Given A:Mat[s,b,c], B:Mat[a,t,c], C:Mat[a,b,c]
#  Solve: X:Mat[a,s], Y:Mat[t,b]
#
#    XA + BY = C
#
#   For 1 <= i <= a, 1 <= j <= b, 1 <= k <= c
#   Σ_m X_im A_{mjk} + Σ_n B_{ink} Y_{nj} = C_{ijk}
#   
# Make the Left Hand Side (LHS) into a relation matrix
# of abc rows (labled by (i,j,k)) and sa+tb columns 
# labled by (i,m) disjoint union (n,j)
function sylvester(A,B,C,T=Float64)
sA=size(A)
s=sA[1]
b=sA[2]
c=sA[3]

sB=size(B)
a=sB[1]
t=sB[2]

rels = zeros(T, a*b*c, a*s+t*b).//1 #zero (of Type Rational{T}) matrix of abc rows and sa+tb columns
cst = zeros(T, a*b*c,1).//1  #zero (of Type Ratioonal{T}) row vector of length abc
for i = 1:a
    for j = 1:b
        for k = 1:c
            # Flattend entry of eq (i,j,k)
            eqIndex = i+a*((j-1)+b*(k-1))            

            # Σ_m X_im A_{mjk}
            for m = 1:s
                vecX = i+a*(m-1)
                vecA = m+s*((j-1)+b*(k-1))
                rels[eqIndex, vecX] = A[vecA] #Julia can flatten arrays naturally
            end

            # Σ_n B_{ink} Y_{nj}
            for n = 1:t
                vecY = n+t*(j-1)
                vecB = i+a*((n-1)+t*(k-1))
                rels[eqIndex, a*s+vecY] = B[vecB]
            end

            # Make constants
            cst[eqIndex] = C[i+a*((j-1)+b*(k-1))]
        end
    end
end
#return rels #cst#, rels
# Solve rel*u=cst
#u = Solve(rels, cst)
u=rels\ cst #Another possible solver
#typeof(u) need to retain information on type of u
T
#return u
# Convert solution to pair of matrices (X,Y)
X = zeros(T, a, s).//1
for i = 1:a 
    for m = 1:s 
        X[i,m] = u[i+a*(m-1)]
    end
end

#return X
Y = zeros(T, t, b).//1
for n = 1:t 
    for j = 1:b
        Y[n,j] = u[a*s+(n+t*(j-1))]
    end
end

# Sanity check
# Test that X*A+B*Y=C 

return X,Y #Return as a tuple
end #end of Sylvester function

##########################
# TODO A.1
#  Get this working.   
#  This will become the refernece implementation
#  I.e. what we test against.
#
# TODO A.2
#   Wrap this up in functions.
###########################

###########################
# TODO B. 
#  We should not be looping over m, n, 
#  but instead copy that whole strip using an array copy.
#
#  Later we we do smarter things.
###########################

#############################
# TODO C 
#
# Make a version to solve 
#  A+BY=C^Z
# XA+B =C^Z
# XA+BY=C^Z
#
# Later we are more clever than 
# having 3 solvers.
#############################