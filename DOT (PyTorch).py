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

import torch

def DOT(values):
    array = torch.cat((-values.flip(0), values))

    matrix = torch.zeros((array.size(0), array.size(0)))

    Emat = torch.tile(array, (array.size(0)>>1, 1))
    Omat = torch.tile(array, (array.size(0)>>1, 1))

    for i in range(array.size(0)>>1):
        Emat[i] = torch.pow(array, (i<<1))
        Omat[i] = torch.pow(array, (i<<1)+1)
    #print(Emat)
    #print(Omat)

    Etemp = torch.zeros((array.size(0)>>1, array.size(0)>>1))
    Otemp = torch.zeros((array.size(0)>>1, array.size(0)>>1))
    #print(Etemp)
    #print(Otemp)

    for i in range(array.size(0)>>1):
        matrix[(i<<1)] = Emat[i]
        matrix[(i<<1)+1] = Omat[i]

        if i > 0:
            matrix[(i<<1)] += (torch.linalg.inv(Etemp[:i,:i]) @ -Etemp[:i,i]).t() @ Emat[:i]
            matrix[(i<<1)+1] += (torch.linalg.inv(Otemp[:i,:i]) @ -Otemp[:i,i]).t() @ Omat[:i]

        matrix[(i<<1)] = torch.nn.functional.normalize(matrix[(i<<1)], dim=-1)
        matrix[(i<<1)+1] = torch.nn.functional.normalize(matrix[(i<<1)+1], dim=-1)

        Etemp[i] = (matrix[(i<<1)] @ Emat.t())
        Otemp[i] = (matrix[(i<<1)+1] @ Omat.t())

    return matrix

#DOT([1.0]) # 2x2 DTT
#DOT([1.0, 3.0]) # 4x4 DTT
matrix = DOT(torch.tensor([1.0, 3.0, 5.0, 7.0])) # 8x8 DTT
#DOT([2.0*i+1.0 for i in range(8)]) # 16x16 DTT
#DOT([2.0*i+1.0 for i in range(16)]) # 32x32 DTT
#DOT([2.0*i+1.0 for i in range(32)]) # 64x64 DTT

print(matrix)
