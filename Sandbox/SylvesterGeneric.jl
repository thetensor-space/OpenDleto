
#
#    A demonstration of a Sylvester solver.
# 
#    Copyright 2021 Amaury Miniño and James B. Wilson.
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
function Sylvester(A,B,C)
T=Int64 #Enter the Type for the ring/field
sA=size(A)# A is (I,J,K)
s=sA[1]
b=sA[2]
c=sA[3]

sB=size(B)
t=sB[2]

rels = zeros(T, a*b*c, a*s+t*b) #zero (of Type T) matrix of abc rows and sa+tb columns
cst = zeros(T, a*b*c, 1)  #zero (of Type T) row vector of length abc
for i = 1:a
    for j = 1:b
        for k = 1:c
            # Flattend entry of eq (i,j,k)
            eqIndex = i+a*(j+b*k)            

            # Σ_m X_im A_{mjk}
            for m = 1:s
                vecX = i+a*m
                vecA = m+s*(j+b*k)
                rels[eqIndex, vecX] = A[vecA]
            end

            # Σ_n B_{ink} Y_{nj}
            for n = 1:t
                vecY = n+t*j
                vecB = i+a*(n+t*k)
                rels[eqIndex, vecY] = B[vecB]
            end

            # Make constants
            cst = C[i+a*(j+b*k)]
        end
    end
end

# Solve rel*u=cst
u = Solve(rels, cst)
#u=\(rels, cst) #Another possible solver

# Convert solution to pair of matrices (X,Y)
X = zeros(T, a, s)
for i = 1:a 
    for m = 1:s 
        X[i,m] = u[i+a*m]
    end
end
Y = zeros(T, t, b)
for n = 1:t 
    for j = 1:t
        Y[n,j] = u[a*s+(n+t*j)]
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