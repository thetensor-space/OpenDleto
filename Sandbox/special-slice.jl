function specialslice(A,d,i,L)
    #Given an array A, return the element for the e'th slice at location p
    #L gives the value for all other indices
    x=selectdim(A,d,i)
    if d>1
        K=
        return specialslice(x,d-1,i, K)
    else 
        return 
    return A[l]
end