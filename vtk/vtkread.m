function StructOut = vtkread(filename)
% VTKREAD Reads VTK into a 3D Matlab array.
% Only implementation performed was for rectilinear grid data types.
% Output is a struct containing the fields found by reading the file.
%  Fernando Zigunov, 2023
    
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



    if ~exist(filename, 'file')
        error('This file does not exist.');
    end

    readCharCount = 5e6;
    
    fid = fopen(filename, 'r','b'); %Reads big-endian files
    readData = 0;
    nextField = '';
    nextFieldType = 'float';
    isNextFieldScalar = 0;
    nextFieldLength = 0;    

    StructOut = struct;

    while (~feof(fid))
        if readData
            %Presizes result
            if isNextFieldScalar
                StructOut.(nextField) = zeros(StructOut.DIMENSIONS);
            else
                StructOut.(nextField) = zeros(nextFieldLength,1);
            end
            %Ready to read the data in the next line
            charsRead = 0;
            while(charsRead < nextFieldLength)
                %Reads next set of characters. Checks if this is the last set
                remChars = nextFieldLength - charsRead;
                readNextChars = min(remChars, readCharCount);
                S = fread(fid, readNextChars, nextFieldType);
                StructOut.(nextField)((charsRead+1):(charsRead+readNextChars)) = S;
                charsRead = charsRead + readNextChars;
                
                disp(['Reading Field ' nextField ':' num2str(charsRead) '/' num2str(nextFieldLength) '(' num2str(100*charsRead/nextFieldLength,3) '%)' '...']);
            end
            readData = 0;
        else
            %Still reading human-readable strings
            S = fgetl(fid);
            try
                S = split(S, ' ');
            catch
                S = {''}; %For some reason the end of file character looks like a number and kills the "split" function
            end
            if contains(upper(S{1}),'X_COORDINATES')
                nextField = 'X_COORDINATES';
                nextFieldType = lower(S{3});
                nextFieldLength = str2double(S{2});
                readData = 1;
                isNextFieldScalar = 0;
            elseif contains(upper(S{1}),'Y_COORDINATES')
                nextField = 'Y_COORDINATES';
                nextFieldType = lower(S{3});
                nextFieldLength = str2double(S{2});
                readData = 1;
                isNextFieldScalar = 0;
            elseif contains(upper(S{1}),'Z_COORDINATES')
                nextField = 'Z_COORDINATES';
                nextFieldType = lower(S{3});
                nextFieldLength = str2double(S{2});
                readData = 1;
                isNextFieldScalar = 0;
            elseif contains(upper(S{1}),'DIMENSIONS')
                StructOut.DIMENSIONS = [str2double(S{2}) str2double(S{3}) str2double(S{4})];
                readData = 0;
            elseif contains(upper(S{1}),'POINT_DATA')
                StructOut.N_POINTS = str2double(S{2});
                nextFieldLength = str2double(S{2});
                readData = 0;
            elseif contains(upper(S{1}),'SCALARS')
                nextField = S{2};
                nextFieldType = lower(S{3});
                readData = 0;
                isNextFieldScalar = 1;
            elseif contains(upper(S{1}),'LOOKUP')
                readData = 1;
            elseif contains(upper(S{1}),'DATASET')
                if ~strcmpi(S{2},'RECTILINEAR_GRID')
                    error('This data set does not contain a rectilinear grid. Other types of grids not yet implemented.');
                end
            end 

        end
    end

    fclose(fid)

end
