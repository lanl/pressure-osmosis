% © 2024. Syracuse University.
% © 2024. Triad National Security, LLC. All rights reserved.
% This program was produced under U.S. Government contract
% 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
% operated by Triad National Security, LLC for the U.S. Department of
% Energy/National Nuclear Security Administration. All rights in the
% program are reserved by Triad National Security, LLC, and the U.S.
% Department of Energy/National Nuclear Security Administration. The
% Government is granted for itself and others acting on its behalf a
% nonexclusive, paid-up, irrevocable worldwide license in this material
% to reproduce, prepare. derivative works, distribute copies to the
% public, perform publicly and display publicly, and to permit
% others to do so.
%
% This program is free software: you can redistribute it and/or modify it
% under the terms of the GNU General Public License as published by the
% Free Software Foundation, either version 3 of the License, or (at your
% option) any later version.
%
% This program is distributed in the hope that it will be useful, but
% WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
% General Public License for more details.
%
% You should have received a copy of the GNU General Public License along
% with this program. If not, see <https://www.gnu.org/licenses/>.

%Minimal Working Example code for usage example
clear; clc; close all;

Nx=10; Ny=20; Nz=30;
Sx=rand(Nx,Ny,Nz);
Sy=rand(Nx,Ny,Nz);
Sz=rand(Nx,Ny,Nz);
delta=[1 2 3];

opts.SolverToleranceRel=1e-4;
opts.SolverToleranceAbs=1e-4;
opts.SolverDevice='GPU';
opts.Verbose=1;
opts.Kernel='cell-centered';
%opts.Kernel='face-crossing';

[P, CGS]=OSMODI(single(Sx),single(Sy),single(Sz),single(delta),opts);

sum(P(:)-single(Sz(:)))


