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


clear; clc; close all;

%Generates a Taylor vortex according to Charonko 2010 DOI 10.1088/0957-0233/21/10/105401
%Taylor vortex used to generate source terms Sx, Sy, Sz
%3D version

t=0.1; %Time, s
H=1e-6; %Vortex strength, m^2
nu=1e-6; %Kinematic Viscosity, m^2/s
rho=1000; %Density, kg/m^3

L0=sqrt(H); %Characteristic Length, m
U0=nu/sqrt(H); %Characteristic Velocity, m/s
T0=L0/U0; %Characteristic Time, s
P0=rho*U0^2; %Characteristic Pressure, Pa

NptsX=101; NptsY=101; NptsZ=101;

x=linspace(-2*L0,2*L0,NptsX); y=linspace(-2*L0,2*L0,NptsY); z=linspace(-2*L0,2*L0,NptsZ);
dx=x(2)-x(1); dy=y(2)-y(1); dz=z(2)-z(1);

[X,Y,Z]=ndgrid(x,y,z);
R=sqrt(X.^2+Y.^2);
Theta=atan2(Y,X);

UR=zeros(size(X));
UTheta=(H/(8*pi)).*(R/(nu*t^2)).*exp(-(R.^2)./(4*nu*t));

P=-(rho*H^2)/(64*(pi^2)*nu*t^3).*exp(-(R.^2)./(2*nu*t));
NormPmax = ((H^2)/(64*(pi^2)*nu*t^3))/(U0^2);

U=UR.*cos(Theta) - UTheta.*sin(Theta);
V=UR.*sin(Theta) + UTheta.*cos(Theta);
W=zeros(size(U));

dudt=zeros(size(U)); dvdt=zeros(size(U)); dwdt=zeros(size(U)); 
[DUDX, DUDY, DUDZ]=gradient(pagetranspose(U),dx,dy,dz); DUDX=pagetranspose(DUDX); DUDY=pagetranspose(DUDY); DUDZ=pagetranspose(DUDZ);
[DVDX, DVDY, DVDZ]=gradient(pagetranspose(V),dx,dy,dz);  DVDX=pagetranspose(DVDX); DVDY=pagetranspose(DVDY); DVDZ=pagetranspose(DVDZ);

%======Generates the Source Terms here======
Sx=(-rho*(dudt+U.*DUDX+V.*DUDY))/(P0/L0);
Sy=(-rho*(dvdt+U.*DVDX+V.*DVDY))/(P0/L0);
Sz=zeros(size(Sx));


%% Exports to vtk
disp('Writing VTK...')
vtkwrite('Source.vtk','RECTILINEAR_GRID',x/L0,y/L0,z/L0,'VECTORS','SOURCE',Sx, Sy, Sz,'BINARY');

%% Runs OmniCUDA with system() [Put the OmniCUDA_OneShot.exe file in this folder]
disp('Firing OmniCUDA_OneShot...')
system('OmniCUDA_OneShot.exe'); %[Put the OmniCUDA_OneShot.exe file in this folder]

%% Compares to the solution by the solver
disp('Reading VTK...')
solverFile='Pressure.vtk';

VTK=vtkread(solverFile);

VTK.PRESSURE=VTK.PRESSURE-VTK.PRESSURE(2,2,2); %Removes corner value 

figure;
subplot(1,2,1)
imagesc(VTK.PRESSURE(:,:,round(NptsZ/2))); colorbar; title('Solver'); daspect([1 1 1])
subplot(1,2,2)
imagesc(P(:,:,round(NptsZ/2))/(rho*U0^2)); colorbar; title('Truth'); daspect([1 1 1])

figure;
subplot(1,2,1)
imagesc(squeeze(VTK.PRESSURE(:,round(NptsY/2),:))); colorbar; title('Solver'); daspect([1 1 1])
subplot(1,2,2)
imagesc(squeeze(P(:,round(NptsY/2),:)/(rho*U0^2))); colorbar; title('Truth'); daspect([1 1 1])


Ptruth=P/(rho*U0^2);
Err=(VTK.PRESSURE - Ptruth)./NormPmax;

ErrorPercent=100*sqrt(sum(Err(:).^2)./numel(Err));
disp(['Error = ' num2str(ErrorPercent, '%0.3f') '%'])





