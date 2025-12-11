function createTensorFromIncidence3(M::AbstractMatrix{T}) where T
    # Check that M is a 01 matrix
    if !all(x -> x == 0 || x == 1, M)
        throw(ArgumentError("Matrix M must be a 0-1 matrix"))
    end
    
    d = size(M, 2)  # number of columns
    e = size(M, 1)  # number of rows
    
    # Check that each row has exactly 3 ones
    for row in 1:e
        if sum(M[row, :]) != 3
            throw(ArgumentError("Each row must have exactly 3 ones"))
        end
    end
    
    # Create d x d x d tensor initialized to zeros
    t = zeros(Float16, d, d, d)
    
    # For each row in M, find the three nonzero columns and set t[i,j,k] = 1
    for row in 1:e
        nonzero_cols = findall(x -> x == 1, M[row, :])
        if length(nonzero_cols) == 3
            i, j, k = nonzero_cols
            t[i, j, k] = 1
            t[i, k, j] = 1
            t[j, i, k] = 1
            t[j, k, i] = 1
            t[k, i, j] = 1
            t[k, j, i] = 1
        end
    end
    
    return t
end


function createTensorFromIncidence(M::AbstractMatrix{T}, m::Integer; field::Type=Float16) where T
    # Check that M is a 0-1 matrix
    if !all(x -> x == 0 || x == 1, M)
        throw(ArgumentError("Matrix M must be a 0-1 matrix"))
    end
    if m < 1
        throw(ArgumentError("m must be >= 1"))
    end

    d = size(M, 2)  # number of columns
    e = size(M, 1)  # number of rows
    if m > d
        throw(ArgumentError("m cannot exceed number of columns"))
    end

    # Create an m-way tensor of size d x d x ... x d with specified element type
    dims = ntuple(_ -> d, m)
    t = zeros(field, dims...)

    # recursive helper to set all permutations of indices in tensor to one(field)
    function set_permutations!(Tarr, inds::Vector{Int}, i::Int)
        if i == length(inds)
            Tarr[Tuple(inds)...] = one(field)
            return
        end
        for j in i:length(inds)
            inds[i], inds[j] = inds[j], inds[i]
            set_permutations!(Tarr, inds, i + 1)
            inds[i], inds[j] = inds[j], inds[i]
        end
    end

    # Process rows that have exactly m ones
    for row in 1:e
        cols = findall(x -> x == 1, M[row, :])
        if length(cols) == m
            set_permutations!(t, collect(cols), 1)
        end
    end

    return t
end

function concatIncidenceMatrices(mats::AbstractMatrix...) 
    if length(mats) == 0
        throw(ArgumentError("At least one incidence matrix is required"))
    end
    ncols = size(mats[1], 2)
    for M in mats
        if ndims(M) != 2
            throw(ArgumentError("All inputs must be 2D matrices"))
        end
        if size(M, 2) != ncols
            throw(ArgumentError("All incidence matrices must have the same number of columns (vertices)"))
        end
        if !all(x -> x == 0 || x == 1, M)
            throw(ArgumentError("Incidence matrices must be 0-1 matrices"))
        end
    end
    return vcat(mats...)
end

# Convenience method accepting a vector/collection of matrices
function concatIncidenceMatrices(coll::AbstractVector{<:AbstractMatrix})
    return concatIncidenceMatrices(coll...)
end


"""
generateHypergraph(kind::Symbol, d::Integer, k::Integer; kwargs...)

Generate standard hypergraph incidence matrices (rows = edges, columns = vertices).

Supported kinds:
- :complete  -> complete k-uniform hypergraph (all k-subsets)
- :random    -> random k-uniform hypergraph; provide either p (probability) or m (number of edges)
- :cycle     -> cyclic k-uniform hypergraph (edges are k consecutive vertices modulo d)
- :chain     -> chain of k-uniform hyperedges connected by single vertices (consecutive edges share exactly one vertex); use num_edges kwarg to specify exact count
- :star      -> star centered at `center` (default 1): all edges contain the center plus any k-1 others
- :fano      -> Fano plane (only valid for d==7, k==3)

Returns a 0-1 Matrix{Int} with one row per edge and d columns.
"""
function generateHypergraph(kind::Symbol, d::Integer, k::Integer; kwargs...)
    if d < 1 || k < 1 || k > d
        throw(ArgumentError("Invalid d,k: require 1 ≤ k ≤ d"))
    end

    # build incidence matrix from list of edges (each edge is a Vector{Int}). There are E edges and d vertices.
    function incidence_from_edges(edges::Vector{Vector{Int}}, d::Integer)
        E = length(edges)
        M = zeros(Int, E, d)
        for (r, edge) in enumerate(edges)
            for v in edge
                if v < 1 || v > d
                    throw(ArgumentError("vertex index out of range: $v"))
                end
                M[r, v] = 1
            end
        end
        return M
    end

    # generate all k-combinations of 1:d
    function combinations_vec(n::Integer, k::Integer)
        res = Vector{Vector{Int}}()
        if k == 0
            push!(res, Int[])
            return res
        end
        buf = Vector{Int}(undef, k)
        function rec(start::Int, depth::Int)
            if depth > k
                push!(res, copy(buf))
                return
            end
            remaining = k - depth + 1
            last = n - remaining + 1
            for i in start:last
                buf[depth] = i
                rec(i + 1, depth + 1)
            end
        end
        rec(1, 1)
        return res
    end

    kind = Symbol(kind)

    if kind === :complete
        edges = combinations_vec(d, k)
        return incidence_from_edges(edges, d)
    elseif kind === :random
        p = get(kwargs, :p, nothing)
        m = get(kwargs, :m, nothing)
        # generate all combinations then sample according to p or m
        all_edges = combinations_vec(d, k)
        if p !== nothing
            if !(0.0 <= p <= 1.0)
                throw(ArgumentError(":p must be in [0,1]"))
            end
            chosen = Vector{Vector{Int}}()
            for e in all_edges
                if rand() < p
                    push!(chosen, e)
                end
            end
            return incidence_from_edges(chosen, d)
        elseif m !== nothing
            if m < 0
                throw(ArgumentError(":m must be non-negative"))
            end
            E = length(all_edges)
            msel = min(Int(m), E)
            idx = randperm(E)[1:msel]
            chosen = all_edges[idx]
            return incidence_from_edges(chosen, d)
        else
            throw(ArgumentError("For :random provide either p=Probability or m=NumEdges"))
        end
    elseif kind === :cycle
        edges = Vector{Vector{Int}}()
        for i in 1:d
            edge = [( (i - 1 + j) % d ) + 1 for j in 0:k-1]
            push!(edges, edge)
        end
        return incidence_from_edges(edges, d)
    elseif kind === :star
        center = get(kwargs, :center, 1)
        if center < 1 || center > d
            throw(ArgumentError("center must be between 1 and d"))
        end
        others = [v for v in 1:d if v != center]
        # all (k-1)-subsets of others
        sub = combinations_vec(d - 1, k - 1)
        # map subsets of positions 1..(d-1) to actual vertex labels in `others`
        edges = Vector{Vector{Int}}()
        for s in sub
            edge = [center]
            for pos in s
                push!(edge, others[pos])
            end
            push!(edges, sort(edge))
        end
        return incidence_from_edges(edges, d)
    elseif kind === :chain
        # Create a chain of k-uniform hyperedges connected by single vertices
        # Each edge shares exactly one vertex with the next edge
        if k < 2
            throw(ArgumentError("chain requires k >= 2"))
        end
        edges = Vector{Vector{Int}}()
        
        # Determine the number of edges to create
        num_edges_kwarg = get(kwargs, :num_edges, nothing)
        if num_edges_kwarg !== nothing
            num_edges = Int(num_edges_kwarg)
            if num_edges < 1
                throw(ArgumentError(":num_edges must be >= 1"))
            end
            # Check if we have enough vertices for the requested number of edges
            # Total vertices needed: k + (num_edges - 1) * (k - 1)
            vertices_needed = k + (num_edges - 1) * (k - 1)
            if vertices_needed > d
                throw(ArgumentError("Not enough vertices for :num_edges=$num_edges with k=$k; need at least $vertices_needed vertices"))
            end
        else
            # Calculate how many edges we can create with d vertices
            # Each edge uses k vertices; each new edge shares 1 with previous, so adds k-1 new vertices
            # Total vertices needed: k + (num_edges - 1) * (k - 1)
            num_edges = div(d - k, k - 1) + 1
        end
        
        if num_edges < 1
            throw(ArgumentError("Not enough vertices to form even one edge; require d >= k"))
        end
        
        vertex_idx = 1
        for edge_num in 1:num_edges
            edge = collect(vertex_idx:(vertex_idx + k - 1))
            push!(edges, edge)
            # Next edge shares the last vertex, so advance by k-1
            vertex_idx += k - 1
        end
        
        return incidence_from_edges(edges, d)
    elseif kind === :fano
        if d != 7 || k != 3
            throw(ArgumentError(":fano only valid for d==7, k==3"))
        end
        blocks = [
            [1,2,3],
            [1,5,6],
            [1,4,7],
            [2,4,6],
            [2,5,7],
            [3,4,5],
            [3,6,7]
        ]
        return incidence_from_edges(blocks, d)
    else
        throw(ArgumentError("Unknown hypergraph kind: $kind"))
    end
end
