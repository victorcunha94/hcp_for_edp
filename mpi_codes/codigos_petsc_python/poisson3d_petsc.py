#!/usr/bin/env python3
"""
Poisson 3D - versão Python (petsc4py) corrigida
"""

import sys
import petsc4py
petsc4py.init(sys.argv)
import argparse
from petsc4py import PETSc
import numpy as np


def compute_matrix(ksp, A, P):
    """Compute the matrix for the 3D Poisson problem (A and optionally P)."""
    da = ksp.getDM()
    mx, my, mz = da.getSizes()
    hx = 1.0 / (mx - 1)
    hy = 1.0 / (my - 1)
    hz = 1.0 / (mz - 1)

    HxHydHz = hx * hy / hz
    HxHzdHy = hx * hz / hy
    HyHzdHx = hy * hz / hx

    # getRanges returns (xs, xe), (ys, ye), (zs, ze) where xe, ye, ze are end indices (exclusive)
    (xs, xe), (ys, ye), (zs, ze) = da.getRanges()

    # Set values on A (and P if provided)
    mat = A
    for k in range(zs, ze):
        for j in range(ys, ye):
            for i in range(xs, xe):
                row = PETSc.Mat.Stencil(i=i, j=j, k=k)

                if i == 0 or j == 0 or k == 0 or i == mx - 1 or j == my - 1 or k == mz - 1:
                    diag = 2.0 * (HxHydHz + HxHzdHy + HyHzdHx)
                    # use setValuesStencil with a single row and single column (the row itself)
                    mat.setValuesStencil(row, [row], [diag])
                else:
                    cols = [
                        PETSc.Mat.Stencil(i=i,   j=j,   k=k-1),  # bottom
                        PETSc.Mat.Stencil(i=i,   j=j-1, k=k),    # south
                        PETSc.Mat.Stencil(i=i-1, j=j,   k=k),    # west
                        PETSc.Mat.Stencil(i=i,   j=j,   k=k),    # center
                        PETSc.Mat.Stencil(i=i+1, j=j,   k=k),    # east
                        PETSc.Mat.Stencil(i=i,   j=j+1, k=k),    # north
                        PETSc.Mat.Stencil(i=i,   j=j,   k=k+1)   # top
                    ]
                    values = [
                        -HxHydHz,     # bottom (k-1)
                        -HxHzdHy,     # south (j-1)
                        -HyHzdHx,     # west (i-1)
                         2.0 * (HxHydHz + HxHzdHy + HyHzdHx),  # center
                        -HyHzdHx,     # east (i+1)
                        -HxHzdHy,     # north (j+1)
                        -HxHydHz      # top (k+1)
                    ]
                    mat.setValuesStencil(row, cols, values)

    # assemble matrix
    mat.assemble()


def compute_rhs(ksp, b):
    """Compute the right-hand side for the 3D Poisson problem"""
    da = ksp.getDM()
    mx, my, mz = da.getSizes()
    hx = 1.0 / (mx - 1)
    hy = 1.0 / (my - 1)
    hz = 1.0 / (mz - 1)

    HxHydHz = hx * hy / hz
    HxHzdHy = hx * hz / hy
    HyHzdHx = hy * hz / hx

    (xs, xe), (ys, ye), (zs, ze) = da.getRanges()
    # da.getVecArray typically returns an array with ordering [k, j, i] (same as DMDA C access barray[k][j][i])
    barr = da.getVecArray(b)

    for k in range(zs, ze):
        for j in range(ys, ye):
            for i in range(xs, xe):
                if i == 0 or j == 0 or k == 0 or i == mx - 1 or j == my - 1 or k == mz - 1:
                    barr[k, j, i] = 2.0 * (HxHydHz + HxHzdHy + HyHzdHx)
                else:
                    barr[k, j, i] = hx * hy * hz

    # garantir montagem (por segurança)
    b.assemble()


def main():
    parser = argparse.ArgumentParser(description="3D Poisson solver with PETSc")
    parser.add_argument("-nx", type=int, default=32)
    parser.add_argument("-ny", type=int, default=32)
    parser.add_argument("-nz", type=int, default=32)
    args, _ = parser.parse_known_args()

    nx, ny, nz = args.nx, args.ny, args.nz

    ksp = PETSc.KSP().create()

    da = PETSc.DMDA().create(
        dim=3,
        sizes=[nx, ny, nz],
        dof=1,
        stencil_width=1,
        boundary_type=[PETSc.DMDA.BoundaryType.NONE] * 3,
        stencil_type=PETSc.DMDA.StencilType.STAR,
        comm=PETSc.COMM_WORLD
    )

    ksp.setDM(da)
    
    # CORREÇÃO: Use as funções diretamente sem o parâmetro ctx
    ksp.setComputeRHS(compute_rhs)
    ksp.setComputeOperators(compute_matrix)
    
    ksp.setFromOptions()
    
    # Crie os vetores b e x *antes* de chamá-los em ksp.solve()
    b = da.createGlobalVec()
    x = da.createGlobalVec()

    # Define o chute inicial para zero
    x.set(0.0)
    
    ksp.solve(b, x)

    # A partir daqui, você deve usar ksp.getSolution() e ksp.getRHS() 
    # para garantir que está usando os vetores *após* a solução, 
    # embora o PETSc já tenha preenchido 'x' e 'b'.
    x = ksp.getSolution()
    b = ksp.getRHS()
    A = ksp.getOperators()[0]

    r = b.duplicate()
    A.mult(x, r)
    r.axpy(-1.0, b)
    PETSc.Sys.Print(f"Residual norm {r.norm(PETSc.NormType.NORM_2)}")

    r.destroy()
    ksp.destroy()
    da.destroy()


if __name__ == "__main__":
    main()
