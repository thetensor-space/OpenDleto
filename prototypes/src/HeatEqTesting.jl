using JuAFEM, SparseArrays

grid = generate_grid(Quadrilateral, (10, 10));
xsize=9
ysize=9

dim = 2
ip = Lagrange{dim, RefCube, 1}()
qr = QuadratureRule{dim, RefCube}(2)
cellvalues = CellScalarValues(qr, ip);

dh = DofHandler(grid)
push!(dh, :u, 1)
close!(dh);

K = create_sparsity_pattern(dh);
M = create_sparsity_pattern(dh);

f = zeros(ndofs(dh));

dt = 0.1
T = 1

ch = ConstraintHandler(dh);

dOmega1 = union(getfaceset.((grid, ), ["left", "right","top", "bottom"])...)
dbc = Dirichlet(:u, dOmega1, (x, t) -> 0)
add!(ch, dbc);

close!(ch)
update!(ch, 0.0);

function doassemble_K!(K::SparseMatrixCSC, f::Vector, cellvalues::CellScalarValues{dim}, dh::DofHandler) where {dim}

    n_basefuncs = getnbasefunctions(cellvalues)
    Ke = zeros(n_basefuncs, n_basefuncs)
    fe = zeros(n_basefuncs)

    assembler = start_assemble(K, f)

    @inbounds for cell in CellIterator(dh)

        fill!(Ke, 0)
        fill!(fe, 0)

        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            dOmega = getdetJdV(cellvalues, q_point)

            for i in 1:n_basefuncs
                v  = shape_value(cellvalues, q_point, i)
                gv = shape_gradient(cellvalues, q_point, i)
                fe[i] += v * dOmega
                for j in 1:n_basefuncs
                    gu = shape_gradient(cellvalues, q_point, j)
                    Ke[i, j] += (gv ⋅ gu) * dOmega
                end
            end
        end

        assemble!(assembler, celldofs(cell), fe, Ke)
    end
    return K, f
end

function doassemble_M!(M::SparseMatrixCSC, cellvalues::CellScalarValues{dim}, dh::DofHandler) where {dim}

    n_basefuncs = getnbasefunctions(cellvalues)
    Me = zeros(n_basefuncs, n_basefuncs)

    assembler = start_assemble(M)

    @inbounds for cell in CellIterator(dh)

        fill!(Me, 0)

        reinit!(cellvalues, cell)

        for q_point in 1:getnquadpoints(cellvalues)
            dOmega = getdetJdV(cellvalues, q_point)

            for i in 1:n_basefuncs
                v  = shape_value(cellvalues, q_point, i)
                for j in 1:n_basefuncs
                    u = shape_value(cellvalues, q_point, j)
                    Me[i, j] += (v ⋅ u) * dOmega
                end
            end
        end

        assemble!(assembler, celldofs(cell), Me)
    end
    return M
end


# the vector is arranged in a strange way need to be twisted
function twistMatrix( m )
	xsize = size(m)[1] - 2
	xsize = size(m)[2] - 2
	res = copy(m)
	firstcols = reshape(m[1:xsize+2, 1:2],(2,xsize+2))'
	for i=1:(xsize+1)
		res[i,1] = firstcols[i,1]
		res[i,2] = firstcols[i,2]
	end
	res[2,1] = firstcols[2,2]
	res[2,2] = firstcols[2,1]
	firstrows = res[1:2,1:(ysize+2)]
	for j=3:(ysize+2)
		res[1,j] = firstrows[2,j]
		res[2,j] = firstrows[1,j]
	end
	return res
end

function untwistMatrix( m )
	xsize = size(m)[1] - 2
	xsize = size(m)[2] - 2
	res = copy(m)
	firstcols = reshape(m[1:xsize+2, 1:2]',(xsize+2,2))
	for i=1:(xsize+1)
		res[i,1] = firstcols[i,1]
		res[i,2] = firstcols[i,2]
	end
	res[3,1] = firstcols[4,1]
	res[4,1] = firstcols[3,1]
	firstrows = res[1:2,1:(ysize+2)]
	for j=3:(ysize+2)
		res[1,j] = firstrows[2,j]
		res[2,j] = firstrows[1,j]
	end
	return res
end


K, f = doassemble_K!(K, f, cellvalues, dh)
M = doassemble_M!(M, cellvalues, dh)
A = (dt .* K) + M;

rhsdata = get_rhs_data(ch, A);

un = zeros(length(f));
for j=1:ysize
	for i = 1:xsize 
		un[j*(xsize+2) + 1+  i] = i* (xsize+1-i)* (xsize+1-2*i) *j *(ysize+1-j)*(xsize+1-2*j)
	end
end

display(reshape(un,(xsize+2,ysize+2)) )

apply!(A, ch);


pvd = paraview_collection("transient-heat.pvd");

for t in 0:dt:T
    #First of all, we need to update the Dirichlet boundary condition values.
    update!(ch, t)

    #Secondly, we compute the right-hand-side of the problem.
    b = dt .* f .+ M * un
    #Then, we can apply the boundary conditions of the current time step.
    apply_rhs!(rhsdata, b, ch)

    #Finally, we can solve the time step and save the solution afterwards.
    u = A \ b;

    vtk_grid("transient-heat-$t", dh) do vtk
        vtk_point_data(vtk, dh, u)
        vtk_save(vtk)
        pvd[t] = vtk
    end
   #At the end of the time loop, we set the previous solution to the current one and go to the next time step.
   un .= u
   Mat=reshape(un,(xsize+2,ysize+2))
   display(Mat)
   display(twistMatrix(Mat))
   display(untwistMatrix(twistMatrix(Mat)))
end

vtk_save(pvd);