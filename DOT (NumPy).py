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

    Emat = numpy.empty([len(array)>>1, len(array)])
    Omat = numpy.empty([len(array)>>1, len(array)])
    for i in range(len(array)>>1):
        Emat[i] = numpy.power(array, (i<<1))
        Omat[i] = numpy.power(array, (i<<1)+1)
    Etemp = numpy.empty([len(array)>>1, len(array)>>1])
    Otemp = numpy.empty([len(array)>>1, len(array)>>1])

    mat = numpy.empty([len(array), len(array)])

    for i in range(len(array)>>1):
        mat[(i<<1)] = Emat[i]
        mat[(i<<1)+1] = Omat[i]

        if i > 0:
            mat[(i<<1)] += numpy.dot(numpy.linalg.inv(Etemp[:i,:i]) @ -Etemp[:i,i], Emat[:i])
            mat[(i<<1)+1] += numpy.dot(numpy.linalg.inv(Otemp[:i,:i]) @ -Otemp[:i,i], Omat[:i])

        mat[(i<<1)] /= numpy.linalg.norm(mat[(i<<1)])
        mat[(i<<1)+1] /= numpy.linalg.norm(mat[(i<<1)+1])

        Etemp[i] = numpy.dot(Emat, mat[(i<<1)])
        Otemp[i] = numpy.dot(Omat, mat[(i<<1)+1])

    return mat

mat = DOT([1, 3]) # 8x8 DTT
#mat = DOT([numpy.cos(7*numpy.pi/16), numpy.cos(5*numpy.pi/16), numpy.cos(3*numpy.pi/16), numpy.cos(numpy.pi/16)]) # 8x8 DCT
print(numpy.allclose(mat @ mat.transpose(), numpy.identity(mat[0].size))) #verification
print(mat)
