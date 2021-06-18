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

torch.set_default_dtype(torch.double)

def DOT(values):
    array = torch.cat((-values.flip(0), values))

    Emat = torch.empty([len(array)>>1, len(array)])
    Omat = torch.empty([len(array)>>1, len(array)])
    for i in range(len(array)>>1):
        Emat[i] = torch.pow(array, (i<<1))
        Omat[i] = torch.pow(array, (i<<1)+1)
    Etemp = torch.empty([len(array)>>1, len(array)>>1])
    Otemp = torch.empty([len(array)>>1, len(array)>>1])

    mat = torch.empty([len(array), len(array)])

    for i in range(len(array)>>1):
        mat[(i<<1)] = Emat[i]
        mat[(i<<1)+1] = Omat[i]

        if i > 0:
            mat[(i<<1)] += torch.tensordot(torch.linalg.inv(Etemp[:i,:i]) @ -Etemp[:i,i], Emat[:i], dims=1)
            mat[(i<<1)+1] += torch.tensordot(torch.linalg.inv(Otemp[:i,:i]) @ -Otemp[:i,i], Omat[:i], dims=1)

        mat[(i<<1)] = torch.nn.functional.normalize(mat[(i<<1)], dim=-1)
        mat[(i<<1)+1] = torch.nn.functional.normalize(mat[(i<<1)+1], dim=-1)

        Etemp[i] = torch.tensordot(Emat, mat[(i<<1)], dims=1)
        Otemp[i] = torch.tensordot(Omat, mat[(i<<1)+1], dims=1)

    return mat

#mat = DOT(torch.tensor([1.0])) # 8x8 DTT
#mat = DOT(torch.tensor([1.0, 3.0])) # 8x8 DTT
mat = DOT(torch.tensor([1.0, 3.0, 5.0, 7.0])) # 8x8 DTT
#DOT([2.0*i+1.0 for i in range(8)]) # 16x16 DTT
#DOT([2.0*i+1.0 for i in range(16)]) # 32x32 DTT
#DOT([2.0*i+1.0 for i in range(32)]) # 64x64 DTT

print(torch.allclose(mat @ mat.t(), torch.eye(mat.size(0))))  #verification
print(mat)
