# Chisel Testing.

import Pkg;
Pkg.activate(".")

using ITensors
using Dleto

frame = [ Index(6, "a$i") for i in 1:3 ];
c = Chisel( [ 1.0 0.0 -1.0; 0.0 1.0 0.0 ], frame );

uc3 = UniversalChisel(frame);
println(uc3)

chisel = CentroidChisel(frame, [true, true, true]);
println(chisel)