'''
Research algorithm implementation for research article
"A General Method for Generating Discrete Orthogonal Matrices" 

Version 1.0
(c) Copyright 2021 Ka-Hou Chan <chankahou (at) ipm.edu.mo>

The algorithm implementation is free software: you can
redistribute it and/or modify it under the terms of the GNU General
Public License as published by the Free Software Foundation,
either version 3 of the License, or (at your option) any later version.

The algorithm implementation is distributed in the hope that it
will be useful, but WITHOUT ANY WARRANTY; without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU General Public License for more details.

You should have received a copy of the GNU General Public
License along with the Kon package.  If not, see
<http://www.gnu.org/licenses/>.
'''

import numpy

def DOT(values):
    array = numpy.concatenate((-numpy.flip(values), values))

    matrix = numpy.zeros((array.size, array.size))
    #print(matrix.shape)

    Ematrix = numpy.tile(array, (array.size>>1, 1))
    Omatrix = numpy.tile(array, (array.size>>1, 1))
    for i in range(array.size>>1):
        Ematrix[i] **= (i<<1)
        Omatrix[i] **= (i<<1)+1
    #print(Ematrix)
    #print(Omatrix)

    Etemp = numpy.zeros((array.size>>1, array.size>>1))
    Otemp = numpy.zeros((array.size>>1, array.size>>1))
    #print(Etemp)
    #print(Otemp)

    for i in range(array.size>>1):
        matrix[(i<<1)] = Ematrix[i]
        matrix[(i<<1)+1] = Omatrix[i]
        if i > 0:
            matrix[(i<<1)] += numpy.matmul(numpy.transpose(numpy.matmul(numpy.linalg.inv(Etemp[:i,:i]), -Etemp[:i,i])), Ematrix[:i])
            matrix[(i<<1)+1] += numpy.matmul(numpy.transpose(numpy.matmul(numpy.linalg.inv(Otemp[:i,:i]), -Otemp[:i,i])), Omatrix[:i])

        matrix[(i<<1)] /= numpy.linalg.norm(matrix[(i<<1)])
        matrix[(i<<1)+1] /= numpy.linalg.norm(matrix[(i<<1)+1])
    
        Etemp[i] = numpy.matmul(matrix[(i<<1)], Ematrix.transpose())
        Otemp[i] = numpy.matmul(matrix[(i<<1)+1], Omatrix.transpose())

    for vector in matrix:
        print(vector*64*numpy.sqrt(array.size))

    print(numpy.allclose(numpy.matmul(matrix, matrix.transpose()), numpy.identity(array.size))) #verification

#DOT([1]) # 2x2 DTT
#DOT([1, 3]) # 4x4 DTT
DOT([1, 3, 5, 7]) # 8x8 DTT
#DOT([2*i+1 for i in range(8)]) # 16x16 DTT
#DOT([2*i+1 for i in range(16)]) # 32x32 DTT
#DOT([2*i+1 for i in range(32)]) # 64x64 DTT
