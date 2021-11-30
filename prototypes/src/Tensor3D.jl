########################################################################
##  CC-BY 2021 Peter A. Brooksbank, Martin Kassabov, James B. Wilson
##
## MIT License
########################################################################



function boxSizeColor( val, norm;factor=1)
    d = 0.5
    n = sqrt(abs(val)/norm) * factor
    #n = round(d*n, digits=4)
    #n = 0.5
    return "255 0 0", [d*n,d*n,d*n], [n,n,n]
end


function rstring(x)
    return string(round(x, digits=4))
end

function boxCornersPLY(  x,y,z, val, norm;factor=1 )
    xwidth = 1; ywidth = 1; zwidth = 1;

    s = ""
    col, shift, widths = boxSizeColor(val, norm;factor)
    # 0 0 0
    s *= rstring(xwidth*x-shift[1]) * " " * rstring(ywidth*y-shift[2]) * " " * rstring(zwidth*z-shift[3]) * " " * col * "\n"
    # 0 0 1
    s *= rstring(xwidth*x-shift[1]) * " " * rstring(ywidth*y-shift[2]) * " " * rstring(zwidth*(z+widths[3])-shift[3]) * " " * col * "\n"
    # 0 1 1
    s *= rstring(xwidth*x-shift[1]) * " " * rstring(ywidth*(y+widths[2])-shift[2]) * " " * rstring(zwidth*(z+widths[3])-shift[3]) * " " * col * "\n"
    # 0 1 0
    s *= rstring(xwidth*x-shift[1]) * " " * rstring(ywidth*(y+widths[2])-shift[2]) * " " * rstring(zwidth*z-shift[3]) * " " * col * "\n"
    # 1 0 0
    s *= rstring(xwidth*(x+widths[1])-shift[1]) * " " * rstring(ywidth*y-shift[2]) * " " * rstring(zwidth*z-shift[3]) * " " * col * "\n"
    # 1 0 1
    s *= rstring(xwidth*(x+widths[1])-shift[1]) * " " * rstring(ywidth*y-shift[2]) * " " * rstring(zwidth*(z+widths[3])-shift[3]) * " " * col * "\n"
    # 1 1 1
    s *= rstring(xwidth*(x+widths[1])-shift[1]) * " " * rstring(ywidth*(y+widths[2])-shift[2]) * " " * rstring(zwidth*(z+widths[3])-shift[3]) * " " * col * "\n"
    # 1 1 0
    s *= rstring(xwidth*(x+widths[1])-shift[1]) * " " * rstring(ywidth*(y+widths[2])-shift[2]) * " " * rstring(zwidth*z-shift[3]) * " " * col * "\n"
    return s
end

function boxFacesPLY( num )
    s = ""
    # 4 verts: 0 1 2 3                 
    s *= string(4) * " " * string(0+8*num) * " " * string(1+8*num) * " " * string(2+8*num) * " " * string(3+8*num) * "\n"
    # 4 verts: 7 6 5 4
    s *= string(4) * " " * string(7+8*num) * " " * string(6+8*num) * " " * string(5+8*num) * " " * string(4+8*num) * "\n"
    # 4 verts: 0 4 5 1
    s *= string(4) * " " * string(0+8*num) * " " * string(4+8*num) * " " * string(5+8*num) * " " * string(1+8*num) * "\n"
    # 4 verts: 1 5 6 2
    s *= string(4) * " " * string(1+8*num) * " " * string(5+8*num) * " " * string(6+8*num) * " " * string(2+8*num) * "\n"
    # 4 verts: 2 6 7 3
    s *= string(4) * " " * string(2+8*num) * " " * string(6+8*num) * " " * string(7+8*num) * " " * string(3+8*num) * "\n"
    # 4 verts: 3 7 4 0
    s *= string(4) * " " * string(3+8*num) * " " * string(7+8*num) * " " * string(4+8*num) * " " * string(0+8*num) * "\n"

    return s
end

function boxRAW(x,y,z, val, norm,ratio)
	s= string(x) * " " * string(y) * " " * string(z) * " 0 " * string(trunc(Int,ratio *val/norm )) * "\n"
	return s
end 


function headerPLY(numvert,numface;title="Visualization of 3-tensor")
    ply = "ply\n"
    ply *= "comment "* title *"\n"
    ply *= "format ascii 1.0\n"
	ply *= "comment vertex description\n"
    ply *= "element vertex " * string(numvert) * "\n"
    ply *= "property float x\n"
    ply *= "property float y\n"
    ply *= "property float z\n"
    ply *= "property uchar red\n"
    ply *= "property uchar green\n"
    ply *= "property uchar blue\n"
	ply *= "comment face description\n"
    ply *= "element face " * string(numface) * "\n"
    ply *= "property list uint8 int32 vertex_index\n"
	ply *= "comment edge description\n"
    ply *= "end_header\n"
	return ply
end

function headerRAW(numvert,numface)
	raw = "x y z meta size\n"
	return raw
end

function tensor3DRAWPLY(t,ratio;factor=1)
	norm = tensorMaxEntry(t)
    verts = ""
    faces = ""
#    verts = "comment Vertices\n"
#    faces = "comment Faces\n"
	raw = ""
    count = 0
    for i = axes(t,1)
		vertst = ""
#		vertst = "comment Starting Slice: " * string(i) *", -, - \n" 
		facest = ""
		rawt = ""
        for j = axes(t,2)
			vertstt = ""
#			vertstt = "comment Starting Slice:" * string(i) * ", " * string(j) *", - \n"
			facestt = ""
			rawtt = "" 
            for k = axes(t,3)
                val = abs(t[i,j,k])
                if (val > norm/ ratio) 
                    vertstt *= boxCornersPLY(i,j,k, val, norm;factor)
                    facestt *= boxFacesPLY(count)
					rawtt *= boxRAW(i,j,k,val, norm,ratio)
                    count += 1
                end
            end
			vertst *= vertstt
			facest *= facestt
			rawt *= rawtt
        end 
		verts *= vertst
		faces *= facest
		raw *= rawt
    end 

    ply = headerPLY(8*count,6*count;title="Visualization of 3-tensor")
    ply *= verts
    ply *= faces
    return ply, headerRAW(8*count,6*count) * raw
end 

# Save a 3-tensor as PLY and RAW files
# factor makes cubes in the output larger
# ratio the ration for largest/smallest cube to print
function save3D(t, nameply,nameraw,ratio;factor=1)
	ply, raw = tensor3DRAWPLY(t,ratio;factor)
    fileply = open(nameply, "w")
    write(fileply, ply)
    close(fileply)
    fileraw = open(nameraw, "w")
    write(fileraw, raw)
    close(fileraw)
    return nothing
end



# produce a ply for a graph of function
# the function is from [0,1]x[0,1] to [0,1]
# xsize, ysize, zsize are scalling parameters
# name is the file name to save
# example of a function:
# s(x,y) = abs(((1-y)^2.5 - x^2.5 + 0.5))^0.4* sign((1-y)^2.5 - x^2.5 + 0.5)

function save3Dgraph(fun,xsize,ysize,zsize,step,name)
	m = zeros(Int64,(step+1,step+1))
	spoints = ""
	sfaces = ""
	pcount = 0
	fcount = 0
	for i = 1:(step+1)
		s = ""
		for j = 1:(step+1)
			z = fun( (i-1)/step, (j-1)/step )
			if ((z >= 0) && (z <= 1))
				pcount +=1
				m[i,j]= pcount
				#generate point
				s *= rstring(1+(i-1)*(xsize-1)/step) * " " * rstring(1+(j-1)*(ysize-1)/step) * " " * rstring(1+z*(zsize-1)) * " 255 0 0 \n"
			end
		end
		spoints *= s
	end
	for i = 1:step
		s = "" 
		for j = 1:step
			if (m[i,j] > 0)  && (m[i+1,j] > 0 ) && (m[i+1,j+1] > 0 ) && (m[i,j+1] > 0 )
				fcount +=1
				#generate face
				s *= "4 " * string(m[i,j]-1) * " " * string(m[i+1,j]-1) * " " * string(m[i+1,j+1]-1) * " " * string(m[i,j+1]-1) * " \n"
			else
				if (m[i,j] > 0)  && (m[i+1,j] > 0 ) && (m[i+1,j+1] > 0 )
					fcount +=1
					#generate face
					s *= "3 " * string(m[i,j]-1) * " " * string(m[i+1,j]-1) * " " * string(m[i+1,j+1]-1) * " \n"
				end
				if (m[i,j] > 0)  && (m[i+1,j] > 0 ) && (m[i,j+1] > 0 )
					fcount +=1
					#generate face
					s *= "3 " * string(m[i,j]-1) * " " * string(m[i+1,j]-1) * " " * string(m[i,j+1]-1) * " \n"
				end
				if (m[i,j] > 0)  && (m[i+1,j+1] > 0 ) && (m[i,j+1] > 0 )
					fcount +=1
					#generate face
					s *= "3 " * string(m[i,j]-1) * " " * string(m[i+1,j+1]-1) * " " * string(m[i,j+1]-1) * " \n"
				end
				if (m[i+1,j] > 0)  && (m[i+1,j+1] > 0 ) && (m[i,j+1] > 0 )
					fcount +=1
					#generate face
					s *= "3 " * string(m[i+1,j]-1) * " " * string(m[i+1,j+1]-1) * " " * string(m[i,j+1]-1) * " \n"
				end
			end
		end
		sfaces *= s
	end
	#combine
	ply = headerPLY(pcount,fcount;title="Graph of a function")

    ply *= spoints
    ply *= sfaces

    file = open(name, "w")
    write(file, ply)
    close(file)
    return nothing
end

nothing