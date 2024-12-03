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

%This code provides a sample data set from a tripped boundary layer at
%2.6m/s (10 frames only, acquired at 5,000 fps) and calculates the
%instantaneous pressure using our OSMODI solver.

%Here we show the pre-processing steps to obtain the pressure gradient
%terms from the Navier Stokes momentum equations.


clear; clc; close all;
load('TRPIV_BL_Sample.mat')

%% Computes derivatives

[dUdx, dUdy, dUdt]=gradient(pagetranspose(U),dx,dy,dt); %Gradient uses meshgrid but our data is in NDgrid
dUdx=pagetranspose(dUdx); dUdy=pagetranspose(dUdy); dUdt=pagetranspose(dUdt); %need to undo pagetranspose

[dVdx, dVdy, dVdt]=gradient(pagetranspose(V),dx,dy,dt); %Gradient uses meshgrid but our data is in NDgrid
dVdx=pagetranspose(dVdx); dVdy=pagetranspose(dVdy); dVdt=pagetranspose(dVdt); %need to undo pagetranspose

%If you want to include the viscous term, add these terms:
%Most cases don't require the viscous term. 
%In this case, the viscous term is ~100x less than the material derivative.
mu=18e-6;
[d2Udx2, ~, ~]=gradient(pagetranspose(dUdx),dx,dy,dt); d2Udx2=pagetranspose(d2Udx2);
[~, d2Udy2, ~]=gradient(pagetranspose(dUdy),dx,dy,dt); d2Udy2=pagetranspose(d2Udy2);
[d2Vdx2, ~, ~]=gradient(pagetranspose(dVdx),dx,dy,dt); d2Vdx2=pagetranspose(d2Vdx2);
[~, d2Vdy2, ~]=gradient(pagetranspose(dVdy),dx,dy,dt); d2Vdy2=pagetranspose(d2Vdy2);

%Calculates Source terms from time-resolved Navier-Stokes equations:
rho=1.2; %kg/m3
viscous = 1;
if viscous
     dPdx=-rho*(dUdt + U.*dUdx +V.*dUdy) + mu*(d2Udx2+d2Udy2); 
     dPdy=-rho*(dVdt + U.*dVdx +V.*dVdy) + mu*(d2Vdx2+d2Vdy2);
else
    dPdx=-rho*(dUdt + U.*dUdx +V.*dUdy); 
    dPdy=-rho*(dVdt + U.*dVdx +V.*dVdy); 
end

%Makes sure all the NaNs are matching
nanMask=isnan(dPdx) | isnan(dPdy);
dPdx(nanMask)=nan; dPdy(nanMask)=nan; %Both fields have to have matching nan masks


%% Calculates pressure from here
opts.SolverToleranceRel=1e-4;
opts.SolverToleranceAbs=1e-4;
opts.SolverDevice='GPU';
opts.Kernel='cell-centered';
opts.Verbose=1;

delta=[dx dy];
P=zeros(size(dPdx));

xinf=237; yinf=163;
Uinf=sqrt(mean(U(xinf,yinf,:)).^2+mean(V(xinf,yinf,:)).^2);
qinf=0.5*1.2*Uinf.^2;

tic
for i=1:size(dPdx,3)
    [p, CGS]=OSMODI(dPdx(:,:,i),dPdy(:,:,i), ones(size(dPdy(:,:,i))),delta,opts);
    pmean(i)=nanmean(p(:));
    P(:,:,i)=p;
    disp(i)
end
toc

%Gets average pressure and plots it
Cp=P/qinf;
if 1
    %Option 1: Remove constant mean over time (i.e. allow fluctuations
    %around reference point)
    meanCp=mean(Cp,3);
    Cp=Cp-meanCp(xinf,yinf); 
elseif 0
    %Option 2: Make fluctuating pressure at fixed point equal zero
    CpInf=Cp(xinf,yinf,:);
    Cp=Cp-CpInf; 
end


%% Makes movie of instantaneous pressure
saveVid = 1;
if saveVid
    vid=VideoWriter('P.mp4','MPEG-4');
    vid.Quality=95;
    vid.FrameRate=30;
    open(vid)
end
figure('Color','w','position',[38.6000 131.4000 1.4336e+03 699.2000]);
for i=1:size(U,3)
    subplot(1,2,1)
    imagesc(Cp(:,:,i)');caxis([-0.4 0.4]);
    title(['Pressure, Frame ' num2str(i)]); set(gca,'ydir','normal');
    cb=colorbar; colormap jet; daspect([1 1 1])
    ylabel(cb,'Cp (P/q_\infty)');

    subplot(1,2,2)
    imagesc(U(:,:,i)'/Uinf);caxis([-0.2 1.2]);
    title(['u Velocity, Frame ' num2str(i)]); set(gca,'ydir','normal');
    cb=colorbar; colormap jet; daspect([1 1 1])
    ylabel(cb,'(u/V_\infty)');
    
    drawnow;pause(0.1)
    if saveVid
        writeVideo(vid,getframe(gcf))
    end
end
if saveVid
    close(vid)
end
close all





