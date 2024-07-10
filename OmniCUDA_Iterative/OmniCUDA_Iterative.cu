/*
 * © 2024. Triad National Security, LLC. All rights reserved.
 * This program was produced under U.S. Government contract
 * 89233218CNA000001 for Los Alamos National Laboratory (LANL), which is
 * operated by Triad National Security, LLC for the U.S. Department of
 * Energy/National Nuclear Security Administration. All rights in the 
 * program are reserved by Triad National Security, LLC, and the U.S.
 * Department of Energy/National Nuclear Security Administration. The
 * Government is granted for itself and others acting on its behalf a
 * nonexclusive, paid-up, irrevocable worldwide license in this material
 * to reproduce, prepare. derivative works, distribute copies to the
 * public, perform publicly and display publicly, and to permit
 * others to do so.
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the
 * Free Software Foundation, either version 3 of the License, or (at your
 * option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program. If not, see <https://www.gnu.org/licenses/>.
 */

//==============================================================
//==============================================================
//====================OMNIDIRECTIONAL CUDA MATRIX SOLVER========
//==============================================================
//==============================================================
//Developed by Fernando Zigunov and John Charonko (2023) - Extreme Fluids Group - Los Alamos National Laboratory
//V00 - This is the ITERATIVE solver.

#pragma once

#include <iostream>
#include <filesystem>
#include <regex>
#include <fstream> 
#include <string>
#include <vector>
#include <iterator>
#include <cstdlib>
#include <curand.h>
#include <curand_kernel.h>
#include <math.h>
#include <time.h>
#include <algorithm>
#include <sstream>
#include <iomanip>

using namespace std;
using namespace std::filesystem;

#define PI 3.141592653589793
#define _X 0
#define _Y 1
#define _Z 2

#define ZEROS 0
#define ONES 1

#define CPU 0
#define CPU_MULT 1
#define GPU 2

#define BLOCKDIM_VEC 512

clock_t tic; clock_t toc; double timeTask;
clock_t tic2; clock_t toc2; double timeTask2;

typedef double varfloat; //Change this to "float" for single precision or "double" for double precision. Found that "float" is ~5-7% faster (not really all that much)

// ====================Structure Definitions================ 
#pragma region

template <typename varfloat>
struct varfloat3 {
    varfloat x;
    varfloat y;
    varfloat z;
};

template <typename varfloat>
struct SolverParameters {
    varfloat solverToleranceRel = 1e-4; //Error allowed for the solver
    varfloat solverToleranceAbs = 1e-4; //Error allowed for the solver
    varfloat PressureSolverToleranceRel = 1e-4; //Error allowed for the solver
    int3 BoxGridPoints = { 100, 100, 1 }; //Number of grid points in the box
    string BoxInputFile; //Contains the string with the box data file.
    string BoxOutputFile = "Pressure_<frame>.vtk"; //Contains the string with the box data file.
    varfloat3<varfloat> GridDelta; //Delta value for derivative approximation
    varfloat density = 1; //Density of the fluid [metric, kg/m3]
    long long totalBoxElements; //Tracks the size of the box
    int NumberOfIterations = 1; //Number of iterations for the pressure solver
    int CheckpointIterations = 0; //Defines interval between iterations before a new snapshot is saved
    varfloat OverRelaxationFactor = 1; // Defines the over-relaxation factor to improve convergence
} ; // Structure to define the parameters of the solver; initializes to default values

template <typename varfloat>
struct BoxContents {
    varfloat* xCoords; //Stores the x coordinates
    varfloat* yCoords; //Stores the y coordinates
    varfloat* zCoords; //Stores the z coordinates
    varfloat* SourceFn_Field_X; //Stores the Source Function here
    varfloat* SourceFn_Field_Y; //Stores the Source Function here
    varfloat* SourceFn_Field_Z; //Stores the Source Function here
    int3 BoxGridSize; //Number of grid points in the box
    varfloat3<varfloat> GridDelta; //Delta value for derivative approximation
    long long totalBoxElements; //Tracks the size of the box
} ; // Structure to hold the contents of the 3D box 

template <typename varfloat>
struct BoxWeights {
    varfloat* Wxp; //Stores the Source Function here
    varfloat* Wxm; //Stores the Source Function here
    varfloat* Wyp; //Stores the Source Function here
    varfloat* Wym; //Stores the Source Function here
    varfloat* Wzp; //Stores the Source Function here
    varfloat* Wzm; //Stores the Source Function here
} ; // Structure to hold the precomputed weights

template <typename varfloat>
struct gridCoefficients {
    varfloat cxx;
    varfloat cxy;
    varfloat cxz;
    varfloat cyy;
    varfloat cyz;
    varfloat czz;
    varfloat ctot; //total to use as a denominator (considers all angles)

    varfloat wxx; //w coefficients are normalized by ctot
    varfloat wxy;
    varfloat wxz;
    varfloat wyy;
    varfloat wyz;
    varfloat wzz;
} ; // Structure to hold the grid coefficients for each of the neighboring cells

#pragma endregion

// ====================Helper Functions================ 
#pragma region
template <typename varfloat>
__host__ __device__ inline int isnan2(varfloat x)
{
    //Apparently uses less registers than the original isnan, See:
    //https://stackoverflow.com/questions/33922103/is-isnan2-as-fast-as-testing-equality
    return x != x;
}

void ClockTic() {
    //Starts the clock
    tic = clock();
}
void ClockTic2() {
    //Starts the clock
    tic2 = clock();
}

void ClockToc(string Text) {
    //Stops the clock & Prints result
    toc = clock() - tic;
    timeTask = ((double)toc) / CLOCKS_PER_SEC; // in seconds
    printf("%s %f s\n", Text.c_str(), timeTask);
}

void ClockToc2(string Text) {
    //Stops the clock & Prints result
    toc2 = clock() - tic2;
    timeTask2 = ((double)toc2) / CLOCKS_PER_SEC; // in seconds
    printf("%s %f s\n", Text.c_str(), timeTask2);
}

template <typename varfloat>
__host__ __device__ varfloat Norm(varfloat3<varfloat> V) {
    //Computes the vector norm
    return sqrt(V.x * V.x + V.y * V.y + V.z * V.z);
}

template <typename varfloat>
__host__ __device__ varfloat3<varfloat> SubtractVectors(varfloat3<varfloat> A, varfloat3<varfloat> B) {
    //Computes the vector subtraction A-B    
    return { A.x - B.x, A.y - B.y, A.z - B.z };
}

bool iequals(const string& a, const string& b)
{
    //Adapted from answers in
    //https://stackoverflow.com/questions/11635/case-insensitive-string-comparison-in-c
    return std::equal(a.begin(), a.end(),
        b.begin(), b.end(),
        [](char a, char b) {
            return tolower(a) == tolower(b);
        });
}

vector<string> split(const string &text, char delim) {
    //little helper function to split strings, inspired by
    //https://stackoverflow.com/a/7408245/20827864
     vector<string> wordvec;
     istringstream iss(text+delim);
     string word;
     while (getline(iss, word, delim)) {
         wordvec.push_back(word);
     }
     return wordvec;
 }

bool replace(std::string& str, const std::string& from, const std::string& to) {
    //Common utility function, see similar implementations in
    //https://stackoverflow.com/questions/3418231/replace-part-of-a-string-with-another-string
    //and
    //https://stackoverflow.com/questions/5878775/how-to-find-and-replace-string.
    size_t start_pos = str.find(from);
    if (start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}

void eraseSubStr(std::string& str, const std::string& toErase)
{
    replace(str,toErase,"");
}

void OutputFileName(string& GenericString, int FrameIndex) {
    //Converts the generic input string (i.e., Image<Camera>) into a numbered string (i.e., Image0)
    replace(GenericString, "<Frame>", to_string(FrameIndex));
    replace(GenericString, "<frame>", to_string(FrameIndex));
    return;
}

template <typename varfloat>
bool ParseInputParameterFile(string fileName, SolverParameters<varfloat>& SP) {
    //This function reads the input parameter file, which initializes the solver parameters as well as the file IO.
    //Input file has the following format:
    //PPPPPP XX
    //where PPPPPP is the parameter name, XX is the parameter value. A line break character separates the parameters. 

    ifstream fileRead(fileName, ifstream::in);
    if (!fileRead) {
        printf("Unable to open input file!");
        return false;
    }
    string currentLine;
    int lineNumber = 0;
    while (getline(fileRead, currentLine))
    {
        //Goes through the file, line by line.
        vector<string> words = split(currentLine, ' '); //Splits string into its "words"

        printf("===line #%d: ", lineNumber + 1); //Prints out line for debugging purposes
        cout << currentLine << "===" << endl; //Prints out line for debugging purposes

        //Parses parameters here.
        if (words[0].compare("SP_CGsolverToleranceRel") == 0) {
            printf("Parameter 'SP_CGsolverToleranceRel' identified. \n");
            if (words.size() >= 2) {
                SP.solverToleranceRel = stod(words[1]);
                printf("SolverParameters.SP_CGsolverToleranceRel = %f.\n", SP.solverToleranceRel);
            }
            else {
                printf("Error in line #%d: Not enough arguments (1 argument required). \n", lineNumber + 1);
            }
        }
        else if (words[0].compare("SP_CGsolverToleranceAbs") == 0) {
            printf("Parameter 'SP_CGsolverToleranceAbs' identified. \n");
            if (words.size() >= 2) {
                SP.solverToleranceAbs = stod(words[1]);
                printf("SolverParameters.SP_CGsolverToleranceAbs = %f.\n", SP.solverToleranceAbs);
            }
            else {
                printf("Error in line #%d: Not enough arguments (1 argument required). \n", lineNumber + 1);
            }
        }
        else if (words[0].compare("SP_PressureSolverToleranceRel") == 0) {
            printf("Parameter 'SP_PressureSolverToleranceRel' identified. \n");
            if (words.size() >= 2) {
                SP.PressureSolverToleranceRel = stod(words[1]);
                printf("SolverParameters.SP_PressureSolverToleranceRel = %f.\n", SP.PressureSolverToleranceRel);
            }
            else {
                printf("Error in line #%d: Not enough arguments (1 argument required). \n", lineNumber + 1);
            }
        }
        else if (words[0].compare("SP_CheckpointIterations") == 0) {
            printf("Parameter 'SP_CheckpointIterations' identified. \n");
            if (words.size() >= 2) {
                SP.CheckpointIterations = stoi(words[1]);
                printf("SolverParameters.SP_CheckpointIterations = %f.\n", SP.CheckpointIterations);
            }
            else {
                printf("Error in line #%d: Not enough arguments (1 argument required). \n", lineNumber + 1);
            }
        }
        else if (words[0].compare("SP_BoxOutputFile") == 0) {
            printf("Parameter 'SP_BoxOutputFile' identified. \n");
            if (words.size() >= 2) {
                //Defines the box data file as the entire content of this line to account for potential spaces in the file path
                int pos = 17;
                printf(currentLine.substr(pos, currentLine.length() - pos).c_str()); printf("\n");
                SP.BoxOutputFile = currentLine.substr(pos, currentLine.length() - pos);
                printf("SolverParameters.BoxOutputFile = %s.\n", SP.BoxOutputFile.c_str());
            }
            else {
                printf("Error in line #%d: Not enough arguments (1 argument required). \n", lineNumber + 1);
            }
        }
        else if (words[0].compare("SP_BoxInputFile") == 0) {
            printf("Parameter 'SP_BoxInputFile' identified. \n");
            if (words.size() >= 2) {
                //Defines the box data file as the entire content of this line to account for potential spaces in the file path
                int pos = 16;
                printf(currentLine.substr(pos, currentLine.length() - pos).c_str()); printf("\n");
                SP.BoxInputFile = currentLine.substr(pos, currentLine.length() - pos);
                printf("SolverParameters.BoxInputFile = %s.\n", SP.BoxInputFile.c_str());
            }
            else {
                printf("Error in line #%d: Not enough arguments (1 argument required). \n", lineNumber + 1);
            }
        }
        else if (words[0].compare("SP_NumberOfIterations") == 0) {
            printf("Parameter 'SP_NumberOfIterations' identified. \n");
            if (words.size() >= 2) {
                SP.NumberOfIterations = stoi(words[1]);
                printf("SolverParameters.SP_NumberOfIterations = %f.\n", SP.NumberOfIterations);
            }
            else {
                printf("Error in line #%d: Not enough arguments (1 argument required). \n", lineNumber + 1);
            }
        }
        else if (words[0].compare("SP_OverRelaxation") == 0) {
            printf("Parameter 'SP_OverRelaxation' identified. \n");
            if (words.size() >= 2) {
                SP.OverRelaxationFactor = stod(words[1]);
                printf("SolverParameters.SP_OverRelaxation = %f.\n", SP.OverRelaxationFactor);
            }
            else {
                printf("Error in line #%d: Not enough arguments (1 argument required). \n", lineNumber + 1);
            }
        }
        else {
            //Parameter not programmed
            printf("This parameter was not recognized. Ignoring line #%d. \n", lineNumber + 1);
            //printf("Parameter: %s\n", words[0].c_str());
        }

        lineNumber++;
    }

    return true;
}

float ReverseFloat(const float inFloat) {
    //Reverses byte order
    float retVal;
    char* floatToConvert = (char*)&inFloat;
    char* returnFloat = (char*)&retVal;

    // swap the bytes into a temporary buffer
    returnFloat[0] = floatToConvert[3];
    returnFloat[1] = floatToConvert[2];
    returnFloat[2] = floatToConvert[1];
    returnFloat[3] = floatToConvert[0];

    return retVal;
}

template <typename varfloat>
bool ReadVTK(string fileName, BoxContents<varfloat>& VTK_Contents) {
    //This function will read a VTK file produced by vtkwrite in Matlab. 
    //Limitations: 
    //[1] Uses vtk v2.0 and only supports box-like input data with the same number of grid points in the X, Y and Z directions.  
    //[2] There must be two fields inside the VTK: (1) Emission and (2) IOR. If IOR is not to be simulated, fill IOR with ones.
    //[3] Only supports the BINARY file format
    //[4] Only supports RECTILINEAR_GRID as the dataset format
    //[5] Binary data is big-endian

    //Expected format follows https://kitware.github.io/vtk-examples/site/VTKFileFormats/ (Accessed in Jan 2023)
    //# vtk Datafile Version 2.0
    //BINARY
    //DATASET RECTILINEAR_GRID
    //DIMENSIONS xx yy zz (should be the same, xx=yy=zz)
    //X_COORDINATES xx float
    //#&$^%(@*&%@)(#%*&@#_)(*%&@#_#@*%&@#_%& (i.e., binary data for X_COORDINATES)
    //Y_COORDINATES yy float
    //#&$^%(@*&%@)(#%*&@#_)(*%&@#_#@*%&@#_%& (i.e., binary data for Y_COORDINATES)
    //Z_COORDINATES zz float
    //#&$^%(@*&%@)(#%*&@#_)(*%&@#_#@*%&@#_%& (i.e., binary data for Z_COORDINATES)
    //POINT_DATA xx*yy*zz
    //VECTORS U float
    //LOOKUP_TABLE default
    //#&$^%(@*&%@)(#%*&@#_)(*%&@#_#@*%&@#_%& (i.e., binary data for U (velocity))
    //VECTORS DUDT float
    //LOOKUP_TABLE default
    //#&$^%(@*&%@)(#%*&@#_)(*%&@#_#@*%&@#_%& (i.e., binary data for DUDT (velocity time derivative))

    ifstream fileRead(fileName, std::ios::binary);
    if (!fileRead) {
        printf("Unable to open box data file!\n");
        return false;
    }

    string currentLine;

    //Line 1: File header
    printf("Reading Box File %s: Line [1]... (VTK header)\n", fileName.c_str());
    getline(fileRead, currentLine);
    vector<string> words = split(currentLine, ' '); //Splits string into its "words"
    bool isVTK = false;
    for (int i = 0; i < words.size(); i++) {
        /*if ((words[i].compare("vtk") == 0)){
            isVTK = true;
        }*/
        if (iequals(words[i], "vtk")) {
            isVTK = true;
        }
    }
    if (!isVTK) {
        printf("Incompatible file header! Header must contain the word 'vtk'.");
        return false;
    }
    printf("Compatible header VTK found.\n");

    //Line 2: File description (irrelevant)
    printf("Reading Box File %s: Line [2]... (File Description)\n", fileName.c_str()); getline(fileRead, currentLine);

    //Line 3: Binary/ASCII
    printf("Reading Box File %s: Line [3]... (Binary flag)\n", fileName.c_str());
    getline(fileRead, currentLine);
    if (!iequals(currentLine, "binary")) {
        printf("Binary flag is not set! Line [3] should have the text 'BINARY' in it.");
        return false;
    }
    printf("Compatible BINARY flag found.\n");

    //Line 4: Rectilinear grid flag
    printf("Reading Box File %s: Line [4]... (Rectilinear grid flag)\n", fileName.c_str());
    getline(fileRead, currentLine);
    words = split(currentLine, ' '); //Splits string into its "words"
    if (!(iequals(words[0], "dataset") && iequals(words[1], "rectilinear_grid"))) {
        printf("Rectilinear flag is not set! Line [4] should have the text 'DATASET RECTILINEAR_GRID' in it.");
        return false;
    }
    printf("Compatible RECTILINEAR_GRID flag found.\n");

    //Line 5: Data dimensions
    printf("Reading Box File %s: Line [5]... (Data dimensions)\n", fileName.c_str());
    getline(fileRead, currentLine);
    words = split(currentLine, ' '); //Splits string into its "words"
    int WordSize;
    if (!iequals(words[0], "dimensions")) {
        printf("Line [5] should begin with the text 'DIMENSIONS'.");
        return false;
    }
    int xDim; int yDim; int zDim;
    if (words.size() >= 4) {
        xDim = stoi(words[1]); yDim = stoi(words[2]); zDim = stoi(words[3]);
    }
    else {
        printf("Line [5] needs at least 3 arguments (e.g.:'DIMENSIONS 64 64 64').");
        return false;
    }
    VTK_Contents.BoxGridSize.x = xDim;
    VTK_Contents.BoxGridSize.y = yDim;
    VTK_Contents.BoxGridSize.z = zDim;
    printf("Grid dimensions are correctly identified: %d x %d x %d.\n", xDim, yDim, zDim);
    VTK_Contents.totalBoxElements = (long long)xDim * (long long)yDim * (long long)zDim;

    VTK_Contents.SourceFn_Field_X = (varfloat*)malloc(VTK_Contents.totalBoxElements * sizeof(varfloat));
    VTK_Contents.SourceFn_Field_Y = (varfloat*)malloc(VTK_Contents.totalBoxElements * sizeof(varfloat));
    VTK_Contents.SourceFn_Field_Z = (varfloat*)malloc(VTK_Contents.totalBoxElements * sizeof(varfloat));

    //Preallocates memory for 1D arrays for each coordinate
    VTK_Contents.xCoords = (varfloat*)malloc(VTK_Contents.BoxGridSize.x * sizeof(varfloat));
    VTK_Contents.yCoords = (varfloat*)malloc(VTK_Contents.BoxGridSize.y * sizeof(varfloat));
    VTK_Contents.zCoords = (varfloat*)malloc(VTK_Contents.BoxGridSize.z * sizeof(varfloat));

    //Gets the grid spacings from the coordinates
    //---X coordinates
    printf("Reading Box File %s: (X coordinates)\n", fileName.c_str());
    getline(fileRead, currentLine);
    words = split(currentLine, ' '); //Splits string into its "words"
    if (!iequals(words[0], "x_coordinates")) {
        printf("The next line should begin with the text 'X_COORDINATES'.");
        return false;
    }
    float currentVal; char currentCharVal_BigEndian[4]; //float is 4 chars long
    char currentCharVal_LittleEndian[4];
    for (int xx = 0; xx < xDim; xx++) {
        fileRead.read(currentCharVal_BigEndian, 4); //Reads the file. It is generated as a big-endian 4-byte float
        for (int ii = 0; ii < 4; ii++) {
            currentCharVal_LittleEndian[3 - ii] = currentCharVal_BigEndian[ii]; //converts to Little-endian
        }
        memcpy(&currentVal, &currentCharVal_LittleEndian[0], sizeof(currentVal));
        VTK_Contents.xCoords[xx] = (varfloat) currentVal;
        //printf("%f;", VTK_Contents.xCoords[xx]);
    }
    getline(fileRead, currentLine); //skips remaining contents of the line before going to y-coords

    //---Y coordinates
    printf("Reading Box File %s: (Y coordinates)\n", fileName.c_str());
    getline(fileRead, currentLine);
    words = split(currentLine, ' '); //Splits string into its "words"
    if (!iequals(words[0], "y_coordinates")) {
        printf("The next line should begin with the text 'Y_COORDINATES'.");
        return false;
    }
    for (int yy = 0; yy < yDim; yy++) {
        fileRead.read(currentCharVal_BigEndian, 4); //Reads the file. It is generated as a big-endian 4-byte float
        for (int ii = 0; ii < 4; ii++) {
            currentCharVal_LittleEndian[3 - ii] = currentCharVal_BigEndian[ii]; //converts to Little-endian
        }
        memcpy(&currentVal, &currentCharVal_LittleEndian[0], sizeof(currentVal));
        VTK_Contents.yCoords[yy] = (varfloat)currentVal;
        //printf("%f;", VTK_Contents.yCoords[yy]);
    }
    getline(fileRead, currentLine); //skips remaining contents of the line before going to z-coords

    //---Z coordinates
    printf("Reading Box File %s: (Z coordinates)\n", fileName.c_str());
    getline(fileRead, currentLine);
    words = split(currentLine, ' '); //Splits string into its "words"
    if (!iequals(words[0], "z_coordinates")) {
        printf("The next line should begin with the text 'Z_COORDINATES'.");
        return false;
    }
    for (int zz = 0; zz < zDim; zz++) {
        fileRead.read(currentCharVal_BigEndian, 4); //Reads the file. It is generated as a big-endian 4-byte float
        for (int ii = 0; ii < 4; ii++) {
            currentCharVal_LittleEndian[3 - ii] = currentCharVal_BigEndian[ii]; //converts to Little-endian
        }
        memcpy(&currentVal, &currentCharVal_LittleEndian[0], sizeof(currentVal));
        VTK_Contents.zCoords[zz] = (varfloat)currentVal;
        //printf("%f;", VTK_Contents.zCoords[zz]);
    }

    //Computes the grid spacings for each grid direction
    VTK_Contents.GridDelta.x = VTK_Contents.xCoords[1] - VTK_Contents.xCoords[0];
    VTK_Contents.GridDelta.y = VTK_Contents.yCoords[1] - VTK_Contents.yCoords[0];
    if (zDim == 1) {
        VTK_Contents.GridDelta.z = 0;
    }
    else {
        VTK_Contents.GridDelta.z = VTK_Contents.zCoords[1] - VTK_Contents.zCoords[0];
    }
    printf("\nGrid Deltas = [%f;%f;%f]\n\n", VTK_Contents.GridDelta.x, VTK_Contents.GridDelta.y, VTK_Contents.GridDelta.z);

    if (VTK_Contents.GridDelta.x < 0 || VTK_Contents.GridDelta.y < 0 || VTK_Contents.GridDelta.z < 0) {
        //Negative grid spacing means coordinate system is backwards
        printf("\n~~~~~~~~~~~~~Error!! Grid Deltas must all be positive!~~~~~~~~~~~~~~~");
        printf("\nIf you are getting this error, it means the order of the elements in the x, y or z vectors is backwards. Flip the order of the vectors! Also, don't forget to flip the sign of the derivatives in that axis as well, as this code assumes a right-handed coordinate system.");
        printf("\nAborting.");
        abort();
    }

    //Finds the string VECTORS; 
    float Progress;
    for (int vectorPos = 0; vectorPos < 1; vectorPos++) { //Accommodating for multiple vector fields if required
        //Finds the string VECTORS:
        printf("\n Reading Box File %s: Finding string 'VECTORS'...\n", fileName.c_str());

        bool stringFound = false;
        string currentWord = "";
        char currentChar;
        while (!(stringFound || fileRead.eof())) {
            fileRead.get(currentChar);
            //printf("New Character: %c; ", currentChar);

            currentWord.push_back(currentChar);
            if (currentWord.size() > 7) {
                //Only looks at the last 7 characters to try and form the string VECTORS
                currentWord = currentWord.substr(1, currentWord.size() - 1);
            }

            //printf("Current String: %s; \n", currentWord);        
            if (iequals(currentWord, "vectors")) {
                //Found it!
                stringFound = true;
                int A = fileRead.tellg();
                printf("Found string 'VECTORS' at position: %i; ", A);
            }
        }
        if (fileRead.eof() && (vectorPos == 0)) {
            //VECTORS is not in the file!
            printf("Error. The word 'VECTORS' is not present in the file, so the contents of the box can't be read.\n");
            return false;
        }

        string currentVectorField = "";
        bool foundSpace = false; string currentShar;
        fileRead.get(currentChar);
        while (!foundSpace) {
            fileRead.get(currentChar);
            currentShar = currentChar;

            if (iequals(currentShar, " ")) {
                foundSpace = true;
            }
            else {
                currentVectorField.push_back(currentChar);
            }
        }
        printf("Vector field name: '%s'\n", currentWord.c_str());

        if (iequals(currentVectorField, "source")) {
            printf("Field correctly identified as a valid field.\n");
            printf("Populating %s field now...\n", currentVectorField.c_str());
            getline(fileRead, currentLine); //finishes reading the current line (data type is always float)

            //Now we will read actual data. Includes a for loop for this one.        
            long long idx = 0;
            //float currentVal; char currentCharVal_BigEndian[4]; //float is 4 chars long
            //char currentCharVal_LittleEndian[4];
            if (iequals(currentVectorField, "source")) {
                for (int zz = 0; zz < zDim; zz++) {
                    for (int yy = 0; yy < yDim; yy++) {
                        for (int xx = 0; xx < xDim; xx++) {
                            //idx = zz*(xDim*yDim) + yy*(xDim) + xx;
                            idx = xx + xDim * (yy + yDim * (zz));

                            fileRead.read(currentCharVal_BigEndian, 4); //Reads the file. It is generated as a big-endian 4-byte float
                            for (int ii = 0; ii < 4; ii++) {
                                currentCharVal_LittleEndian[3 - ii] = currentCharVal_BigEndian[ii]; //converts to Little-endian
                            }
                            memcpy(&currentVal, &currentCharVal_LittleEndian[0], sizeof(currentVal));
                            VTK_Contents.SourceFn_Field_X[idx] = (varfloat)currentVal;

                            fileRead.read(currentCharVal_BigEndian, 4); //Reads the file. It is generated as a big-endian 4-byte float
                            for (int ii = 0; ii < 4; ii++) {
                                currentCharVal_LittleEndian[3 - ii] = currentCharVal_BigEndian[ii]; //converts to Little-endian
                            }
                            memcpy(&currentVal, &currentCharVal_LittleEndian[0], sizeof(currentVal));
                            VTK_Contents.SourceFn_Field_Y[idx] = (varfloat)currentVal;

                            fileRead.read(currentCharVal_BigEndian, 4); //Reads the file. It is generated as a big-endian 4-byte float
                            for (int ii = 0; ii < 4; ii++) {
                                currentCharVal_LittleEndian[3 - ii] = currentCharVal_BigEndian[ii]; //converts to Little-endian
                            }
                            memcpy(&currentVal, &currentCharVal_LittleEndian[0], sizeof(currentVal));
                            VTK_Contents.SourceFn_Field_Z[idx] = (varfloat)currentVal;

                            if (isnan(VTK_Contents.SourceFn_Field_X[idx]) || isnan(VTK_Contents.SourceFn_Field_Y[idx]) || isnan(VTK_Contents.SourceFn_Field_Z[idx])) {
                                VTK_Contents.SourceFn_Field_X[idx] = NAN;
                                VTK_Contents.SourceFn_Field_Y[idx] = NAN;
                                VTK_Contents.SourceFn_Field_Z[idx] = NAN;
                            }

                        }
                    }
                    Progress = (((float)vectorPos * (float)VTK_Contents.totalBoxElements) + (float)idx) / (1 * (float)VTK_Contents.totalBoxElements);
                    printf("\33[2K\r Loading SOURCE Field... Progress %.2f %%", Progress * 100.0);
                    fflush(stdout);
                }
            }
        }
        else {
            printf("This vector field name was not recognized. Only 'SOURCE' is identified as a valid field. SOURCE = -RHO*DU/DT where DU/DT is the material derivative.\n");
            return false;
        }

    }

    fileRead.close();
    return true;
}

template <typename varfloat>
bool SaveVTK(string fileName, varfloat* fieldOut, SolverParameters<varfloat> SP, BoxContents<varfloat> VTK_Contents) {
    //Writes a VTK file containing the output pressure field computed by the algorithm.
    ofstream fileWrite(fileName, std::ios::binary);

    if (!fileWrite) {
        printf("Error opening the VTK file for output!\n");
        return false;
    }
    string stringOut = "# vtk DataFile Version 2.0\n"; //header
    fileWrite.write(stringOut.c_str(), stringOut.size() * sizeof(char));

    stringOut = "Pressure Field Computed with Omnidirectional Scheme\n"; //file description
    fileWrite.write(stringOut.c_str(), stringOut.size() * sizeof(char));

    stringOut = "BINARY\n"; //Binary flag
    fileWrite.write(stringOut.c_str(), stringOut.size() * sizeof(char));

    stringOut = "DATASET RECTILINEAR_GRID\n"; //DATASET flag
    fileWrite.write(stringOut.c_str(), stringOut.size() * sizeof(char));

    stringOut = "DIMENSIONS "; stringOut.append(to_string(VTK_Contents.BoxGridSize.x)); stringOut.append(" ");  //Dimensions of data
    stringOut.append(to_string(VTK_Contents.BoxGridSize.y)); stringOut.append(" ");
    stringOut.append(to_string(VTK_Contents.BoxGridSize.z)); stringOut.append("\n");
    fileWrite.write(stringOut.c_str(), stringOut.size() * sizeof(char));

    //----Coordinates---
    float Coord;
    float* CoordListX; CoordListX = new float[VTK_Contents.BoxGridSize.x];
    float* CoordListY; CoordListY = new float[VTK_Contents.BoxGridSize.y];
    float* CoordListZ; CoordListZ = new float[VTK_Contents.BoxGridSize.z];
    for (int i = 0; i < VTK_Contents.BoxGridSize.x; i++) {
        CoordListX[i] = ReverseFloat((float) VTK_Contents.xCoords[i]); //We need to flip the endianness before writing to file
    }
    for (int i = 0; i < VTK_Contents.BoxGridSize.y; i++) {
        CoordListY[i] = ReverseFloat((float)VTK_Contents.yCoords[i]); //We need to flip the endianness before writing to file
    }
    for (int i = 0; i < VTK_Contents.BoxGridSize.z; i++) {
        CoordListZ[i] = ReverseFloat((float)VTK_Contents.zCoords[i]); //We need to flip the endianness before writing to file
    }

    stringOut = "X_COORDINATES "; stringOut.append(to_string(VTK_Contents.BoxGridSize.x)); stringOut.append(" float\n");
    fileWrite.write(stringOut.c_str(), stringOut.size() * sizeof(char));
    fileWrite.write((char*)CoordListX, sizeof(float) * VTK_Contents.BoxGridSize.x); //write the binary data    
    stringOut = "\n"; fileWrite.write(stringOut.c_str(), stringOut.size() * sizeof(char)); //new line character

    stringOut = "Y_COORDINATES "; stringOut.append(to_string(VTK_Contents.BoxGridSize.y)); stringOut.append(" float\n");
    fileWrite.write(stringOut.c_str(), stringOut.size() * sizeof(char));
    fileWrite.write((char*)CoordListY, sizeof(float) * VTK_Contents.BoxGridSize.y); //write the binary data    
    stringOut = "\n"; fileWrite.write(stringOut.c_str(), stringOut.size() * sizeof(char)); //new line character

    stringOut = "Z_COORDINATES "; stringOut.append(to_string(VTK_Contents.BoxGridSize.z)); stringOut.append(" float\n");
    fileWrite.write(stringOut.c_str(), stringOut.size() * sizeof(char));
    fileWrite.write((char*)CoordListZ, sizeof(float) * VTK_Contents.BoxGridSize.z); //write the binary data    
    stringOut = "\n"; fileWrite.write(stringOut.c_str(), stringOut.size() * sizeof(char)); //new line character

    stringOut = "POINT_DATA "; stringOut.append(to_string(VTK_Contents.totalBoxElements)); stringOut.append("\n");
    fileWrite.write(stringOut.c_str(), stringOut.size() * sizeof(char));
    //----Scalar field---
    stringOut = "SCALARS PRESSURE float\n"; fileWrite.write(stringOut.c_str(), stringOut.size() * sizeof(char));
    stringOut = "LOOKUP_TABLE default\n"; fileWrite.write(stringOut.c_str(), stringOut.size() * sizeof(char));

    //We have to flip endianness of the data prior to writing
    float* fieldToSave;
    fieldToSave = (float*)malloc(VTK_Contents.totalBoxElements * sizeof(float));

    for (int i = 0; i < (VTK_Contents.totalBoxElements); i++) {
        fieldToSave[i] = ReverseFloat((float)fieldOut[i]);
    }
    fileWrite.write((char*)fieldToSave, sizeof(float) * VTK_Contents.totalBoxElements); //write the binary data for the 3D field

    fileWrite.close();
    printf("Successfully saved VTK file!\n");
    free(fieldToSave);

    return true;
}

template <typename varfloat>
void gridCoefficients2D(BoxContents<varfloat>& VTK_Contents, gridCoefficients<varfloat>& gridCoeffs) {
    //This function precomputes the coefficients for the grid in 2D (i.e. a trivial case)
    varfloat dx = 1.0;
    varfloat dy = (varfloat) (VTK_Contents.GridDelta.y / VTK_Contents.GridDelta.x); //Preconditioning by normalizing by x

    varfloat dd = sqrt(dx * dx + dy * dy);
    gridCoeffs.cxx = 2 * (dd - dx);
    gridCoeffs.cxy = dx + dy - dd;
    gridCoeffs.cyy = 2 * (dd - dy);
    gridCoeffs.ctot = 2 * (gridCoeffs.cxx + 4 * gridCoeffs.cxy + gridCoeffs.cyy);
    //printf("c=[%f, %f, %f] %f\n", gridCoeffs.cxx, gridCoeffs.cxy, gridCoeffs.cyy, gridCoeffs.ctot);

    gridCoeffs.wxx = gridCoeffs.cxx / gridCoeffs.ctot;
    gridCoeffs.wxy = gridCoeffs.cxy / gridCoeffs.ctot;
    gridCoeffs.wyy = gridCoeffs.cyy / gridCoeffs.ctot;
}

template <typename varfloat>
varfloat computeProjectionConstant(varfloat dx, varfloat dy, varfloat dz, int Npoints, int j, int k) {
    //Here we will compute the projection constant c_jk for a given pair of faces j,k.
    //j and k are values for each of the axes. The constants are defined in the defines, but X=0, Y=1, Z=2
    varfloat dj, dk, dl, dl2;
    varfloat cjk;
    if (j == k) {
        // Far face integral, j = k
        if (j == _X) {
            dj = dx; dl = dy; dl2 = dz;
        }
        else if (j == _Y) {
            dj = dy; dl = dx; dl2 = dz;
        }
        else if (j == _Z) {
            dj = dz; dl = dx; dl2 = dy;
        }

        cjk = 0; //Integrates here
        varfloat prevIntegrand; varfloat nextIntegrand;
        varfloat alpha1 = atan(dj / dl2); varfloat dphi = PI / (2 * (varfloat)Npoints);
        for (varfloat phi = alpha1; phi <= (PI / 2); phi += dphi) {
            varfloat dl1c = dl * cos(phi);
            varfloat dl1s = dl * sin(phi);
            if (phi == alpha1) {
                prevIntegrand = (PI * (dl2 * dl1s - dj * dl1c) + 2 * (dj * dl1c * atan(dj / dl1s) - dl2 * dl1s * atan(dj / dl1s))) * dphi;
            }
            else {
                nextIntegrand = (PI * (dl2 * dl1s - dj * dl1c) + 2 * (dj * dl1c * atan(dj / dl1s) - dl2 * dl1s * atan(dj / dl1s))) * dphi;
                cjk += (prevIntegrand + nextIntegrand) / 2;
                prevIntegrand = nextIntegrand;
            }
        }
    }
    else {
        // Dihedral face integrals
        if (j == _X) {
            dj = dx;
            if (k == _Y) {
                dk = dy; dl = dz;
            }
            else if (k == _Z) {
                dk = dz; dl = dy;
            }
        }
        else if (j == _Y) {
            dj = dy;
            if (k == _X) {
                dk = dx; dl = dz;
            }
            else if (k == _Z) {
                dk = dz; dl = dx;
            }
        }
        else if (j == _Z) {
            dj = dz;
            if (k == _X) {
                dk = dx; dl = dy;
            }
            else if (k == _Y) {
                dk = dy; dl = dx;
            }
        }
        varfloat I_Ac = 0.5 * dl * dl * atanh((dj * dj * dk * dk) / (2 * dl * dl * dl * dl + 2 * dl * dl * dj * dj + 2 * dl * dl * dk * dk + dj * dj * dk * dk)); //Integral A

        varfloat I_Bc = 0;
        varfloat prevIntegrand = 0; varfloat nextIntegrand;
        varfloat alpha1 = atan(dj / dl); varfloat dphi = PI / (2 * (varfloat)Npoints);
        varfloat dl_2 = dl * dl; varfloat dj_2 = dj * dj;
        for (varfloat phi = 0; phi <= (PI / 2); phi += dphi) {
            varfloat dkc = dk * cos(phi); varfloat dkc_2 = dkc * dkc;
            varfloat dks = dk * sin(phi); varfloat dks_2 = dks * dks;

            if (phi <= alpha1) {
                nextIntegrand = (dl * dks * (atan(dl / dkc) - (dl * dkc) / (dl_2 + dkc_2)) + dkc * dks * (log(dkc / sqrt(dl_2 + dkc_2)) + (dl_2) / (2 * (dl_2 + dkc_2)))) * dphi;
            }
            else {
                nextIntegrand = (dl * dks * (atan(dj / dks) - (dj * dks) / (dj_2 + dks_2)) + dkc * dks * (log(dks / sqrt(dj_2 + dks_2)) + (dj_2) / (2 * (dj_2 + dks_2)))) * dphi;
            }
            I_Bc += (prevIntegrand + nextIntegrand) / 2;
            prevIntegrand = nextIntegrand;
        }

        varfloat I_Cc = 0;
        for (varfloat phi = alpha1; phi <= (PI / 2); phi += dphi) {
            if (phi == alpha1) {
                prevIntegrand = 0.5 * (dj * dk * dk * sin(phi) * (2 * dl * sin(phi) - dj * cos(phi)) / (dj * dj + dk * dk * sin(phi) * sin(phi))) * dphi;
            }
            else {
                nextIntegrand = 0.5 * (dj * dk * dk * sin(phi) * (2 * dl * sin(phi) - dj * cos(phi)) / (dj * dj + dk * dk * sin(phi) * sin(phi))) * dphi;
                I_Cc += (prevIntegrand + nextIntegrand) / 2;
                prevIntegrand = nextIntegrand;
            }
        }

        cjk = I_Ac + I_Bc + I_Cc;
    }
    return cjk;
}

template <typename varfloat>
void gridCoefficients3D(BoxContents<varfloat>& VTK_Contents, gridCoefficients<varfloat>& gridCoeffs, int Npoints) {
    //This function precomputes the coefficients for the grid in 3D (a little more involved)
    varfloat dx = 1.0;
    varfloat dy = VTK_Contents.GridDelta.y / VTK_Contents.GridDelta.x; //Preconditioning by normalizing by x
    varfloat dz = VTK_Contents.GridDelta.z / VTK_Contents.GridDelta.x; //Preconditioning by normalizing by x

    if ((dy == 1.0) && (dz == 1.0)) {
        //equal grid spacing, use precomputed constants to save time
        gridCoeffs.cxx = 0.62776840115648713;
        gridCoeffs.cyy = 0.62776840115648713;
        gridCoeffs.czz = 0.62776840115648713;
        gridCoeffs.cxy = 0.62845603826512653;
        gridCoeffs.cxz = 0.62845603826512653;
        gridCoeffs.cyz = 0.62845603826512653;
    }
    else {
        gridCoeffs.cxx = computeProjectionConstant(dx, dy, dz, Npoints, _X, _X);
        gridCoeffs.cxy = computeProjectionConstant(dx, dy, dz, Npoints, _X, _Y);
        gridCoeffs.cxz = computeProjectionConstant(dx, dy, dz, Npoints, _X, _Z);
        gridCoeffs.cyy = computeProjectionConstant(dx, dy, dz, Npoints, _Y, _Y);
        gridCoeffs.cyz = computeProjectionConstant(dx, dy, dz, Npoints, _Y, _Z);
        gridCoeffs.czz = computeProjectionConstant(dx, dy, dz, Npoints, _Z, _Z);
    }

    gridCoeffs.ctot = 2 * (gridCoeffs.cxx + gridCoeffs.cyy + gridCoeffs.czz + 4 * gridCoeffs.cxy + 4 * gridCoeffs.cxz + 4 * gridCoeffs.cyz);
    //printf("c_tot=%f\n", gridCoeffs.ctot);

    varfloat ctot_ideal = 2 * PI * (dx * dy + dx * dz + dy * dz);
    varfloat coeff_rel_error = abs(gridCoeffs.ctot - ctot_ideal) / ctot_ideal;
    //printf("c_err=%e\n", coeff_rel_error);

    gridCoeffs.wxx = gridCoeffs.cxx / gridCoeffs.ctot;
    gridCoeffs.wxy = gridCoeffs.cxy / gridCoeffs.ctot;
    gridCoeffs.wxz = gridCoeffs.cxz / gridCoeffs.ctot;
    gridCoeffs.wyy = gridCoeffs.cyy / gridCoeffs.ctot;
    gridCoeffs.wyz = gridCoeffs.cyz / gridCoeffs.ctot;
    gridCoeffs.wzz = gridCoeffs.czz / gridCoeffs.ctot;
}


template <typename varfloat>
void FillBox(varfloat* boxToFill, int BoxContents, SolverParameters<varfloat> SP) {
    //Fills the 3D box with one of the objects as dictated by BoxContents
    varfloat xLoc, yLoc, zLoc;
    int idxBox;
    //Reference control volumes
    for (int zz = 0; zz < SP.BoxGridPoints.z; zz++) {
        for (int yy = 0; yy < SP.BoxGridPoints.y; yy++) {
            for (int xx = 0; xx < SP.BoxGridPoints.x; xx++) {
                long long idxBox = (long long)xx + (long long)SP.BoxGridPoints.x * ((long long)yy + (long long)SP.BoxGridPoints.y * ((long long)zz));

                if (BoxContents == ZEROS) {
                    boxToFill[idxBox] = 0.0f;
                }
                else if (BoxContents == ONES) {
                    boxToFill[idxBox] = 1.0f;
                }
                else {
                    boxToFill[idxBox] = 0.0f;
                }
            }
        }
    }
}
#pragma endregion

// ====================GPU CUDA Function Kernels================
#pragma region


template <typename varfloat>
__global__ void printScalar_GPU(varfloat* a) {
    //For debugging purposes
    printf("%f\n", *a);
}

template <typename varfloat>
__global__ void printVector1_GPU(varfloat* v) {
    //For debugging purposes
    printf("%e; ", v[0]);
}

template <typename varfloat>
__global__ void printVector2_GPU(varfloat* v) {
    //For debugging purposes
    printf("%e\n", v[49]);
}

template <typename varfloat>
__global__ void addVectors_GPU(varfloat* a, varfloat* b, varfloat* out, SolverParameters<varfloat>* SP) {
    long long idxCenter = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;

    if (idxCenter >= (SP->BoxGridPoints.x * SP->BoxGridPoints.y * SP->BoxGridPoints.z)) {
        //Idles if voxels beyond volume size
        return;
    }
    out[idxCenter] = a[idxCenter] + b[idxCenter];
}

template <typename varfloat>
__global__ void subtractVectors_GPU(varfloat* a, varfloat* b, varfloat* out, SolverParameters<varfloat>* SP) {
    long long idxCenter = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;

    if (idxCenter >= (SP->BoxGridPoints.x * SP->BoxGridPoints.y * SP->BoxGridPoints.z)) {
        //Idles if voxels beyond volume size
        return;
    }
    out[idxCenter] = a[idxCenter] - b[idxCenter];
}

template <typename varfloat>
__global__ void divide(varfloat* num, varfloat* den, varfloat* out) {
    unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_x == 0) {
        *out = *num / *den;
    }
}

template <typename varfloat>
__global__ void vectorDot_GPU_Slow(varfloat* a, varfloat* b, varfloat* out, SolverParameters<varfloat>* SP) {
    //naive slow implementation
    unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    varfloat tmp = 0.0;
    if (index_x == 0) {
        for (long long i = 0; i < (SP->BoxGridPoints.x * SP->BoxGridPoints.y * SP->BoxGridPoints.z); i++) {
            if (!isnan2(a[i]) && !isnan2(b[i])) {
                tmp += a[i] * b[i];
                //printf("i=%lld; a=%f, b=%f, tmp=%f, \n", i, a[i], b[i],tmp);
            }
        }
        *out = tmp;
    }
}

template <typename varfloat>
__global__ void scalarVectorMult_GPU(varfloat* scalar, varfloat* a, varfloat* out, SolverParameters<varfloat>* SP) {
    unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
    if (index_x < (SP->BoxGridPoints.x * SP->BoxGridPoints.y * SP->BoxGridPoints.z)) {
        out[index_x] = a[index_x] * *scalar;
    }
}


template <typename varfloat>
__global__ void vectorDot_GPU(varfloat* a, varfloat* b, varfloat* out, SolverParameters<varfloat>* SP) {
    __shared__ varfloat shared_tmp[BLOCKDIM_VEC];

    long long index = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;
    long long maxIdx = (long long)SP->BoxGridPoints.x * (long long)SP->BoxGridPoints.y * (long long)SP->BoxGridPoints.z;

    if (index == 0) {
        *out = 0.0; //We need to reset the output variable when we start
    }

    if (index < maxIdx) {
        if (!isnan2(a[index]) || !isnan2(b[index])) {
            shared_tmp[threadIdx.x] = a[index] * b[index]; //Dont norm nans
        }
        else {
            shared_tmp[threadIdx.x] = 0.0;//nans become zeros
        }
    }
    else {
        shared_tmp[threadIdx.x] = 0.0;
    }

    __syncthreads();

    // reduction within block
    unsigned int i = blockDim.x / 2;
    while (i != 0) {
        if (threadIdx.x < i) {
            shared_tmp[threadIdx.x] += shared_tmp[threadIdx.x + i];
        }
        __syncthreads();
        i /= 2;
    }

    // atomic add the partial reduction in out
    if (threadIdx.x == 0) {
        atomicAdd(out, shared_tmp[0]);
    }
}

template <typename varfloat>
void Precompute_Weights_Iteration_nPlusOne(BoxWeights<varfloat>& Weights, SolverParameters<varfloat> SP, BoxContents<varfloat> VTK_Contents, gridCoefficients<varfloat> gc) {
    //Precomputes the weights to save time in the GPU computation
    long long GridX = SP.BoxGridPoints.x;
    long long GridY = SP.BoxGridPoints.y;
    long long GridZ = SP.BoxGridPoints.z;

    if (GridZ == 1) {
        //2D Case
        long long zz = 0;
        for (long long yy = 0; yy < GridY; yy++) {
            for (long long xx = 0; xx < GridX; xx++) {
                long long idxCenter = xx + GridX * (yy + GridY * zz);

                long long idx_xpxp = idxCenter + 2;
                long long idx_xmxm = idxCenter - 2;
                long long idx_ypyp = idxCenter + GridX * 2;
                long long idx_ymym = idxCenter - GridX * 2;

                long long idx_xpyp = idxCenter + 1 + GridX;
                long long idx_xpym = idxCenter + 1 - GridX;
                long long idx_xmyp = idxCenter - 1 + GridX;
                long long idx_xmym = idxCenter - 1 - GridX;

                //Computes the boolean values for each index
                varfloat bxpxp = ((xx + 2) >= GridX) || isnan2(VTK_Contents.SourceFn_Field_X[idx_xpxp]);
                varfloat bypyp = ((yy + 2) >= GridY) || isnan2(VTK_Contents.SourceFn_Field_X[idx_ypyp]);
                varfloat bxmxm = ((xx - 2) < 0) || isnan2(VTK_Contents.SourceFn_Field_X[idx_xmxm]);
                varfloat bymym = ((yy - 2) < 0) || isnan2(VTK_Contents.SourceFn_Field_X[idx_ymym]);

                varfloat bxpyp = ((xx + 1) >= GridX) || ((yy + 1) >= GridY) || isnan2(VTK_Contents.SourceFn_Field_X[idx_xpyp]);
                varfloat bxpym = ((xx + 1) >= GridX) || ((yy - 1) < 0) || isnan2(VTK_Contents.SourceFn_Field_X[idx_xpym]);
                varfloat bxmyp = ((yy + 1) >= GridY) || ((xx - 1) < 0) || isnan2(VTK_Contents.SourceFn_Field_X[idx_xmyp]);
                varfloat bxmym = ((xx - 1) < 0) || ((yy - 1) < 0) || isnan2(VTK_Contents.SourceFn_Field_X[idx_xmym]);

                //Computes the weights for the [n] coefficients
                Weights.Wxp[idxCenter] = (1.0 - bxpxp) * gc.wxx + (2.0 - (bxpyp + bxpym)) * gc.wxy;
                Weights.Wxm[idxCenter] = (1.0 - bxmxm) * gc.wxx + (2.0 - (bxmyp + bxmym)) * gc.wxy;
                Weights.Wyp[idxCenter] = (1.0 - bypyp) * gc.wyy + (2.0 - (bxmyp + bxpyp)) * gc.wxy;
                Weights.Wym[idxCenter] = (1.0 - bymym) * gc.wyy + (2.0 - (bxmym + bxpym)) * gc.wxy;
            }
        }

    }
    else {
        //3D case
        for (long long zz = 0; zz < GridZ; zz++) {
            for (long long yy = 0; yy < GridY; yy++) {
                for (long long xx = 0; xx < GridX; xx++) {
                    long long idxCenter = xx + GridX * (yy + GridY * zz);

                    long long idx_xpxp = idxCenter + 2;
                    long long idx_xmxm = idxCenter - 2;
                    long long idx_ypyp = idxCenter + GridX * 2;
                    long long idx_ymym = idxCenter - GridX * 2;
                    long long idx_zpzp = idxCenter + GridX * GridY * 2;
                    long long idx_zmzm = idxCenter - GridX * GridY * 2;

                    long long idx_xpyp = idxCenter + 1 + GridX;
                    long long idx_xpym = idxCenter + 1 - GridX;
                    long long idx_xpzp = idxCenter + 1 + GridX * GridY;
                    long long idx_xpzm = idxCenter + 1 - GridX * GridY;

                    long long idx_xmyp = idxCenter - 1 + GridX;
                    long long idx_xmym = idxCenter - 1 - GridX;
                    long long idx_xmzp = idxCenter - 1 + GridX * GridY;
                    long long idx_xmzm = idxCenter - 1 - GridX * GridY;

                    long long idx_ypzp = idxCenter + GridX + GridX * GridY;
                    long long idx_ypzm = idxCenter + GridX - GridX * GridY;

                    long long idx_ymzp = idxCenter - GridX + GridX * GridY;
                    long long idx_ymzm = idxCenter - GridX - GridX * GridY;

                    varfloat bxpxp = ((xx + 2) >= GridX) || isnan2(VTK_Contents.SourceFn_Field_X[idx_xpxp]);
                    varfloat bxmxm = ((xx - 2) < 0) || isnan2(VTK_Contents.SourceFn_Field_X[idx_xmxm]);
                    varfloat bypyp = ((yy + 2) >= GridY) || isnan2(VTK_Contents.SourceFn_Field_X[idx_ypyp]);
                    varfloat bymym = ((yy - 2) < 0) || isnan2(VTK_Contents.SourceFn_Field_X[idx_ymym]);
                    varfloat bzpzp = ((zz + 2) >= GridZ) || isnan2(VTK_Contents.SourceFn_Field_X[idx_zpzp]);
                    varfloat bzmzm = ((zz - 2) < 0) || isnan2(VTK_Contents.SourceFn_Field_X[idx_zmzm]);

                    varfloat bxpyp = ((xx + 1) >= GridX) || ((yy + 1) >= GridY) || isnan2(VTK_Contents.SourceFn_Field_X[idx_xpyp]);
                    varfloat bxpym = ((xx + 1) >= GridX) || ((yy - 1) < 0) || isnan2(VTK_Contents.SourceFn_Field_X[idx_xpym]);
                    varfloat bxpzp = ((xx + 1) >= GridX) || ((zz + 1) >= GridZ) || isnan2(VTK_Contents.SourceFn_Field_X[idx_xpzp]);
                    varfloat bxpzm = ((xx + 1) >= GridX) || ((zz - 1) < 0) || isnan2(VTK_Contents.SourceFn_Field_X[idx_xpzm]);

                    varfloat bxmyp = ((xx - 1) < 0) || ((yy + 1) >= GridY) || isnan2(VTK_Contents.SourceFn_Field_X[idx_xmyp]);
                    varfloat bxmym = ((xx - 1) < 0) || ((yy - 1) < 0) || isnan2(VTK_Contents.SourceFn_Field_X[idx_xmym]);
                    varfloat bxmzp = ((xx - 1) < 0) || ((zz + 1) >= GridZ) || isnan2(VTK_Contents.SourceFn_Field_X[idx_xmzp]);
                    varfloat bxmzm = ((xx - 1) < 0) || ((zz - 1) < 0) || isnan2(VTK_Contents.SourceFn_Field_X[idx_xmzm]);

                    varfloat bypzp = ((yy + 1) >= GridY) || ((zz + 1) >= GridZ) || isnan2(VTK_Contents.SourceFn_Field_X[idx_ypzp]);
                    varfloat bypzm = ((yy + 1) >= GridY) || ((zz - 1) < 0) || isnan2(VTK_Contents.SourceFn_Field_X[idx_ypzm]);

                    varfloat bymzp = ((yy - 1) < 0) || ((zz + 1) >= GridZ) || isnan2(VTK_Contents.SourceFn_Field_X[idx_ymzp]);
                    varfloat bymzm = ((yy - 1) < 0) || ((zz - 1) < 0) || isnan2(VTK_Contents.SourceFn_Field_X[idx_ymzm]);

                    //Computes the weights for the [n+1] coefficients                    
                    Weights.Wxp[idxCenter] = (1 - bxpxp) * gc.wxx + (2 - (bxpyp + bxpym)) * gc.wxy + (2 - (bxpzp + bxpzm)) * gc.wxz;
                    Weights.Wxm[idxCenter] = (1 - bxmxm) * gc.wxx + (2 - (bxmyp + bxmym)) * gc.wxy + (2 - (bxmzp + bxmzm)) * gc.wxz;
                    Weights.Wyp[idxCenter] = (1 - bypyp) * gc.wyy + (2 - (bxmyp + bxpyp)) * gc.wxy + (2 - (bypzm + bypzp)) * gc.wyz;
                    Weights.Wym[idxCenter] = (1 - bymym) * gc.wyy + (2 - (bxmym + bxpym)) * gc.wxy + (2 - (bymzm + bymzp)) * gc.wyz;
                    Weights.Wzp[idxCenter] = (1 - bzpzp) * gc.wzz + (2 - (bxmzp + bxpzp)) * gc.wxz + (2 - (bymzp + bypzp)) * gc.wyz;
                    Weights.Wzm[idxCenter] = (1 - bzmzm) * gc.wzz + (2 - (bxmzm + bxpzm)) * gc.wxz + (2 - (bymzm + bypzm)) * gc.wyz;
                }
            }
        }
    }

}

template <typename varfloat>
__global__ void MatrixMul_Omnidirectional_GPU(varfloat* Result, varfloat* PressureField, varfloat* RHS, SolverParameters<varfloat>* SP, gridCoefficients<varfloat>* gc, varfloat* Wxp, varfloat* Wxm, varfloat* Wyp, varfloat* Wym, varfloat* Wzp, varfloat* Wzm) {
    //This is the bit of code that performs the matrix multiplication Result=A*x (where A is the weight matrix and x is the PressureField)
    //The RHS of the equation is also provided so we can find the points where we have NAN's
    long long xx = blockIdx.x * blockDim.x + threadIdx.x;
    long long yy = blockIdx.y * blockDim.y + threadIdx.y;
    long long zz = blockIdx.z * blockDim.z + threadIdx.z;

    if ((xx >= SP->BoxGridPoints.x) || (yy >= SP->BoxGridPoints.y) || (zz >= SP->BoxGridPoints.z)) {
        //Idles if voxels beyond volume size
        return;
    }

    if (SP->BoxGridPoints.z == 1) {
        //2D Case
        //Finds the indices for each of the adjacent cells and their neighbors
        long long GridX = (long long)SP->BoxGridPoints.x;
        long long GridY = (long long)SP->BoxGridPoints.y;

        long long idxCenter = xx + GridX * (yy + GridY * zz);

        if ((RHS[idxCenter] != RHS[idxCenter])) {
            //The RHS here is a nan, so simply makes the result at this point a nan as well
            Result[idxCenter] = NAN;
        }
        else {
            long long idx_xp = idxCenter + 1;
            long long idx_xm = idxCenter - 1;
            long long idx_yp = idxCenter + GridX;
            long long idx_ym = idxCenter - GridX;

            varfloat bxp = ((xx + 1) >= GridX) || (RHS[idx_xp] != RHS[idx_xp]); // isnans exposed as inequalities to reduce the number of registers required (from 112 to 56) [i.e. isnan(X) is the same as X!=X]
            varfloat byp = ((yy + 1) >= GridY) || (RHS[idx_yp] != RHS[idx_yp]);
            varfloat bxm = ((xx - 1) < 0) || (RHS[idx_xm] != RHS[idx_xm]);
            varfloat bym = ((yy - 1) < 0) || (RHS[idx_ym] != RHS[idx_ym]);

            //Adds the pressure values to right-hand side for this cell 
            varfloat R = PressureField[idxCenter];
            R -= bxp ? 0.0 : Wxp[idxCenter] * PressureField[idx_xp]; //done this way to prevent access outside allocated memory 
            R -= bxm ? 0.0 : Wxm[idxCenter] * PressureField[idx_xm];
            R -= byp ? 0.0 : Wyp[idxCenter] * PressureField[idx_yp];
            R -= bym ? 0.0 : Wym[idxCenter] * PressureField[idx_ym];
            Result[idxCenter] = R;
        }
    }
    else {
        //3D Case
        //Finds the indices for each of the adjacent cells and their neighbors
        long long GridX = (long long)SP->BoxGridPoints.x;
        long long GridY = (long long)SP->BoxGridPoints.y;
        long long GridZ = (long long)SP->BoxGridPoints.z;

        long long idxCenter = xx + GridX * (yy + GridY * zz);

        if (RHS[idxCenter] != RHS[idxCenter]) {
            //The RHS here is a nan, so simply makes the result at this point a nan as well
            Result[idxCenter] = NAN;
        }
        else {
            long long idx_xp = idxCenter + 1;
            long long idx_xm = idxCenter - 1;
            long long idx_yp = idxCenter + GridX;
            long long idx_ym = idxCenter - GridX;
            long long idx_zp = idxCenter + GridX * GridY;
            long long idx_zm = idxCenter - GridX * GridY;

            varfloat bxp = ((xx + 1) >= GridX) || (RHS[idx_xp] != RHS[idx_xp]); // isnans exposed as inequalities to reduce the number of registers required (from 112 to 80) [i.e. isnan(X) is the same as X!=X]
            varfloat byp = ((yy + 1) >= GridY) || (RHS[idx_yp] != RHS[idx_yp]);
            varfloat bzp = ((zz + 1) >= GridZ) || (RHS[idx_zp] != RHS[idx_zp]);
            varfloat bxm = ((xx - 1) < 0) || (RHS[idx_xm] != RHS[idx_xm]);
            varfloat bym = ((yy - 1) < 0) || (RHS[idx_ym] != RHS[idx_ym]);
            varfloat bzm = ((zz - 1) < 0) || (RHS[idx_zm] != RHS[idx_zm]);

            //Adds the pressure values to right-hand side for this cell 
            varfloat R = PressureField[idxCenter];
            R -= bxp ? 0 : Wxp[idxCenter] * PressureField[idx_xp]; //done this way to prevent access outside allocated memory 
            R -= bxm ? 0 : Wxm[idxCenter] * PressureField[idx_xm];
            R -= byp ? 0 : Wyp[idxCenter] * PressureField[idx_yp];
            R -= bym ? 0 : Wym[idxCenter] * PressureField[idx_ym];
            R -= bzp ? 0 : Wzp[idxCenter] * PressureField[idx_zp];
            R -= bzm ? 0 : Wzm[idxCenter] * PressureField[idx_zm];
            Result[idxCenter] = R;
        }

    }

}

template <typename varfloat>
__global__ void UpdateRHS_Vector_GPU(varfloat* PressureField, varfloat* RHS, varfloat* SourceX, varfloat* SourceY, varfloat* SourceZ, SolverParameters<varfloat>* SP, gridCoefficients<varfloat>* gc, varfloat* Wxp, varfloat* Wxm, varfloat* Wyp, varfloat* Wym, varfloat* Wzp, varfloat* Wzm) {
    //Computes the right-hand side vector based on the values of the pressures for all cells, considering boundaries, etc.
    long long xx = blockIdx.x * blockDim.x + threadIdx.x;
    long long yy = blockIdx.y * blockDim.y + threadIdx.y;
    long long zz = blockIdx.z * blockDim.z + threadIdx.z;

    if ((xx >= SP->BoxGridPoints.x) || (yy >= SP->BoxGridPoints.y) || (zz >= SP->BoxGridPoints.z)) {
        //Idles if voxels beyond volume size
        return;
    }

    if (SP->BoxGridPoints.z == 1) {
        //Finds the indices for each of the adjacent cells and their neighbors
        long long GridX = (long long)SP->BoxGridPoints.x;
        long long GridY = (long long)SP->BoxGridPoints.y;
        long long idxCenter = xx + GridX * (yy + GridY * zz);

        //dx and dy for the grid
        varfloat GridDX = SP->GridDelta.x;
        varfloat GridDY = SP->GridDelta.y;

        if (SourceX[idxCenter] != SourceX[idxCenter]) {
            //The source value here is a nan, so simply makes the RHS at this point a nan as well
            RHS[idxCenter] = NAN;
        }
        else {
            long long idx_xp = idxCenter + 1;
            long long idx_yp = idxCenter + GridX;
            long long idx_xm = idxCenter - 1;
            long long idx_ym = idxCenter - GridX;

            //Computes the boolean values for each index
            varfloat bxp = ((xx + 1) >= GridX) || (SourceX[idx_xp] != SourceX[idx_xp]);
            varfloat byp = ((yy + 1) >= GridY) || (SourceX[idx_yp] != SourceX[idx_yp]);
            varfloat bxm = ((xx - 1) < 0) || (SourceX[idx_xm] != SourceX[idx_xm]);
            varfloat bym = ((yy - 1) < 0) || (SourceX[idx_ym] != SourceX[idx_ym]);

            //Computes the weights for the [n] coefficients
            varfloat wxmax = gc->wxx + 2 * gc->wxy; varfloat wymax = gc->wyy + 2 * gc->wxy; //Weights for out-of-bounds conditions
            varfloat rhs_cx = GridDY / (2.0 * (GridDX + GridDY));
            varfloat rhs_cy = GridDX / (2.0 * (GridDX + GridDY));

            //Adds the pressure values to right-hand side for this cell
            varfloat R = 0.0;
            R += bxp ? 0.0 : ((wxmax - Wxp[idxCenter]) * PressureField[idx_xp] - rhs_cx * (SourceX[idx_xp] + SourceX[idxCenter]) * (GridDX / 2.0));
            R += bxm ? 0.0 : ((wxmax - Wxm[idxCenter]) * PressureField[idx_xm] + rhs_cx * (SourceX[idx_xm] + SourceX[idxCenter]) * (GridDX / 2.0));
            R += byp ? 0.0 : ((wymax - Wyp[idxCenter]) * PressureField[idx_yp] - rhs_cy * (SourceY[idx_yp] + SourceY[idxCenter]) * (GridDY / 2.0));
            R += bym ? 0.0 : ((wymax - Wym[idxCenter]) * PressureField[idx_ym] + rhs_cy * (SourceY[idx_ym] + SourceY[idxCenter]) * (GridDY / 2.0));

            varfloat w_in = wxmax * (bxp + bxm) + wymax * (byp + bym); //Weight for the center coefficient
            R += w_in * PressureField[idxCenter];
            RHS[idxCenter] = R;

        }
    }
    else {
        //3D case
        long long GridX = (long long)SP->BoxGridPoints.x;
        long long GridY = (long long)SP->BoxGridPoints.y;
        long long GridZ = (long long)SP->BoxGridPoints.z;
        long long idxCenter = xx + GridX * (yy + GridY * zz);

        //dx and dy and dz for the grid
        varfloat GridDX = SP->GridDelta.x;
        varfloat GridDY = SP->GridDelta.y;
        varfloat GridDZ = SP->GridDelta.z;

        if (SourceX[idxCenter] != SourceX[idxCenter]) {
            //The source value here is a nan, so simply makes the RHS at this point a nan as well
            RHS[idxCenter] = NAN;
        }
        else {
            long long idx_xp = idxCenter + 1;
            long long idx_xm = idxCenter - 1;
            long long idx_yp = idxCenter + GridX;
            long long idx_ym = idxCenter - GridX;
            long long idx_zp = idxCenter + GridX * GridY;
            long long idx_zm = idxCenter - GridX * GridY;

            //Computes the boolean values for each index
            varfloat bxp = ((xx + 1) >= GridX) || (SourceX[idx_xp] != SourceX[idx_xp]);
            varfloat bxm = ((xx - 1) < 0) || (SourceX[idx_xm] != SourceX[idx_xm]);
            varfloat byp = ((yy + 1) >= GridY) || (SourceX[idx_yp] != SourceX[idx_yp]);
            varfloat bym = ((yy - 1) < 0) || (SourceX[idx_ym] != SourceX[idx_ym]);
            varfloat bzp = ((zz + 1) >= GridZ) || (SourceX[idx_zp] != SourceX[idx_zp]);
            varfloat bzm = ((zz - 1) < 0) || (SourceX[idx_zm] != SourceX[idx_zm]);

            //Computes the weights for the [n] coefficients
            varfloat wxmax = gc->wxx + 2 * gc->wxy + 2 * gc->wxz; varfloat wymax = gc->wyy + 2 * gc->wxy + 2 * gc->wyz; varfloat wzmax = gc->wzz + 2 * gc->wxz + 2 * gc->wyz;//Weights for out-of-bounds conditions
            varfloat rhs_den = 2.0 * (GridDX * GridDY + GridDX * GridDZ + GridDY * GridDZ);
            varfloat rhs_cx = (GridDY * GridDZ) / rhs_den;
            varfloat rhs_cy = (GridDX * GridDZ) / rhs_den;
            varfloat rhs_cz = (GridDX * GridDY) / rhs_den;

            //Adds the pressure values to right-hand side for this cell   
            varfloat R = 0.0;
            R += bxp ? 0.0 : ((wxmax - Wxp[idxCenter]) * PressureField[idx_xp] - rhs_cx * (SourceX[idx_xp] + SourceX[idxCenter]) * GridDX / 2);
            R += bxm ? 0.0 : ((wxmax - Wxm[idxCenter]) * PressureField[idx_xm] + rhs_cx * (SourceX[idx_xm] + SourceX[idxCenter]) * GridDX / 2);
            R += byp ? 0.0 : ((wymax - Wyp[idxCenter]) * PressureField[idx_yp] - rhs_cy * (SourceY[idx_yp] + SourceY[idxCenter]) * GridDY / 2);
            R += bym ? 0.0 : ((wymax - Wym[idxCenter]) * PressureField[idx_ym] + rhs_cy * (SourceY[idx_ym] + SourceY[idxCenter]) * GridDY / 2);
            R += bzp ? 0.0 : ((wzmax - Wzp[idxCenter]) * PressureField[idx_zp] - rhs_cz * (SourceZ[idx_zp] + SourceZ[idxCenter]) * GridDZ / 2);
            R += bzm ? 0.0 : ((wzmax - Wzm[idxCenter]) * PressureField[idx_zm] + rhs_cz * (SourceZ[idx_zm] + SourceZ[idxCenter]) * GridDZ / 2);

            varfloat w_in = wxmax * (bxp + bxm) + wymax * (byp + bym) + wzmax * (bzp + bzm); //Weight for the center coefficient
            R += w_in * PressureField[idxCenter];
            RHS[idxCenter] = R;
        }
    }
}

template <typename varfloat>
void ConjugateGradientSolver_GPU(varfloat* PressureField, varfloat* RHS, BoxWeights<varfloat> Weights, SolverParameters<varfloat> SolverConfig, BoxContents<varfloat> VTK_Contents, gridCoefficients<varfloat> gridCoeffs, string OutputFileName, ofstream* csvOutput) {
    // Allocate GPU memory for source field and pressure field
    //Creates concurrent streams so processing can occur in parallel
    const int nStreams = 4;
    cudaStream_t stream[nStreams];
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    //Memory allocation
    long long boxArraySize = sizeof(varfloat) * VTK_Contents.totalBoxElements;
    varfloat* d_PressureField; varfloat* d_RHS; varfloat* d_PressureFieldOld;
    varfloat* d_rk; varfloat* d_rkp1; varfloat* d_pk; varfloat* d_temp;
    cudaMalloc(&d_PressureField, boxArraySize); cudaMalloc(&d_RHS, boxArraySize); cudaMalloc(&d_PressureFieldOld, boxArraySize);
    cudaMalloc(&d_rk, boxArraySize); cudaMalloc(&d_rkp1, boxArraySize);
    cudaMalloc(&d_pk, boxArraySize); cudaMalloc(&d_temp, boxArraySize);

    cudaMemcpyAsync(d_PressureField, PressureField, boxArraySize, cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync(d_RHS, RHS, boxArraySize, cudaMemcpyHostToDevice, stream[1]);

    //Allocate GPU memory for the weights
    varfloat* d_Wxp; varfloat* d_Wxm; varfloat* d_Wyp; varfloat* d_Wym; varfloat* d_Wzp; varfloat* d_Wzm;
    cudaMalloc(&d_Wxp, boxArraySize); cudaMalloc(&d_Wxm, boxArraySize); cudaMalloc(&d_Wyp, boxArraySize); cudaMalloc(&d_Wym, boxArraySize); cudaMalloc(&d_Wzp, boxArraySize); cudaMalloc(&d_Wzm, boxArraySize);
    cudaMemcpyAsync(d_Wxp, Weights.Wxp, boxArraySize, cudaMemcpyHostToDevice, stream[2]); cudaMemcpyAsync(d_Wxm, Weights.Wxm, boxArraySize, cudaMemcpyHostToDevice, stream[3]);
    cudaMemcpyAsync(d_Wyp, Weights.Wyp, boxArraySize, cudaMemcpyHostToDevice, stream[0]); cudaMemcpyAsync(d_Wym, Weights.Wym, boxArraySize, cudaMemcpyHostToDevice, stream[1]);
    cudaMemcpyAsync(d_Wzp, Weights.Wzp, boxArraySize, cudaMemcpyHostToDevice, stream[2]); cudaMemcpyAsync(d_Wzm, Weights.Wzm, boxArraySize, cudaMemcpyHostToDevice, stream[3]);

    //Allocate GPU memory for the source terms
    varfloat* d_SourceX; varfloat* d_SourceY; varfloat* d_SourceZ;
    cudaMalloc(&d_SourceX, boxArraySize); cudaMalloc(&d_SourceY, boxArraySize); cudaMalloc(&d_SourceZ, boxArraySize);
    cudaMemcpyAsync(d_SourceX, VTK_Contents.SourceFn_Field_X, boxArraySize, cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync(d_SourceY, VTK_Contents.SourceFn_Field_Y, boxArraySize, cudaMemcpyHostToDevice, stream[1]);
    cudaMemcpyAsync(d_SourceZ, VTK_Contents.SourceFn_Field_Z, boxArraySize, cudaMemcpyHostToDevice, stream[2]);

    //Allocates auxiliary variables
    gridCoefficients<varfloat>* d_gridCoeffs;
    cudaMalloc(&d_gridCoeffs, sizeof(gridCoefficients<varfloat>));
    cudaMemcpyAsync(d_gridCoeffs, &gridCoeffs, sizeof(gridCoefficients<varfloat>), cudaMemcpyHostToDevice, stream[0]);

    SolverParameters<varfloat>* d_SolverConfig;
    cudaMalloc(&d_SolverConfig, sizeof(SolverParameters<varfloat>));
    cudaMemcpyAsync(d_SolverConfig, &SolverConfig, sizeof(SolverParameters<varfloat>), cudaMemcpyHostToDevice, stream[1]);

    //Allocates scalars
    varfloat* d_beta; varfloat* d_alpha; varfloat* d_r_norm; varfloat* d_r_norm_old; varfloat* d_temp_scal; varfloat* d_over_relax;
    cudaMalloc((void**)&d_beta, sizeof(varfloat)); cudaMalloc((void**)&d_alpha, sizeof(varfloat)); cudaMalloc((void**)&d_r_norm, sizeof(varfloat));
    cudaMalloc((void**)&d_r_norm_old, sizeof(varfloat)); cudaMalloc((void**)&d_temp_scal, sizeof(varfloat)); cudaMalloc((void**)&d_over_relax, sizeof(varfloat));

    cudaMemcpy(d_over_relax, &SolverConfig.OverRelaxationFactor, sizeof(varfloat), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock3D; dim3 numBlocks3D; //3D for the matrix kernel
    if (VTK_Contents.BoxGridSize.z == 1) {
        threadsPerBlock3D = dim3(16, 16, 1);
        numBlocks3D = dim3(ceil(VTK_Contents.BoxGridSize.x / (varfloat)threadsPerBlock3D.x), ceil(VTK_Contents.BoxGridSize.y / (varfloat)threadsPerBlock3D.y), 1);
    }
    else {
        threadsPerBlock3D = dim3(4, 8, 8);
        numBlocks3D = dim3(ceil(VTK_Contents.BoxGridSize.x / (varfloat)threadsPerBlock3D.x), ceil(VTK_Contents.BoxGridSize.y / (varfloat)threadsPerBlock3D.y), ceil(VTK_Contents.BoxGridSize.z / (varfloat)threadsPerBlock3D.z));
    }
    dim3 threadsPerBlock1D = dim3(BLOCKDIM_VEC, 1, 1); //1D for the vector kernels
    dim3 numBlocks1D = dim3(ceil(VTK_Contents.totalBoxElements / (varfloat)threadsPerBlock1D.x), 1, 1);

    //Outer pressure loop for solver
    varfloat first_r_norm_init = 0.0;
    varfloat p_norm = 1.0; varfloat delta_p_norm = 1.0; //to assess convergence
    for (int pressureIterations = 0; pressureIterations < SolverConfig.NumberOfIterations; pressureIterations++) {
        ClockTic();
        //=====Updates RHS terms=====
        cudaDeviceSynchronize();
        UpdateRHS_Vector_GPU << <numBlocks3D, threadsPerBlock3D, 0, stream[0] >> > (d_PressureField, d_RHS, d_SourceX, d_SourceY, d_SourceZ, d_SolverConfig, d_gridCoeffs, d_Wxp, d_Wxm, d_Wyp, d_Wym, d_Wzp, d_Wzm); //temp=A*x_0

        //=====Starts CG solver computations=====
        cudaDeviceSynchronize();
        MatrixMul_Omnidirectional_GPU << <numBlocks3D, threadsPerBlock3D, 0, stream[0] >> > (d_temp, d_PressureField, d_RHS, d_SolverConfig, d_gridCoeffs, d_Wxp, d_Wxm, d_Wyp, d_Wym, d_Wzp, d_Wzm); //temp=A*x_0
        cudaDeviceSynchronize();
        subtractVectors_GPU << <numBlocks1D, threadsPerBlock1D, 0, stream[0] >> > (d_RHS, d_temp, d_rk, d_SolverConfig); //r_0=b-A*x_0

        cudaDeviceSynchronize();
        cudaMemcpyAsync(d_pk, d_rk, boxArraySize, cudaMemcpyDeviceToDevice, stream[0]); //p_0=r_0
        vectorDot_GPU << <numBlocks1D, threadsPerBlock1D, 0, stream[1] >> > (d_rk, d_rk, d_r_norm_old, d_SolverConfig); //r_k dot r_kvarfloat r_norm; 
        cudaDeviceSynchronize();

        varfloat r_norm_init;
        cudaMemcpy(&r_norm_init, d_r_norm_old, sizeof(varfloat), cudaMemcpyDeviceToHost); // initial residual norm
        r_norm_init = sqrt(r_norm_init);
        if (pressureIterations == 0) {
            first_r_norm_init = r_norm_init;
        }

        printf("Initial Residual Norm=%f\n", r_norm_init);
        for (int cgs_iter = 0; cgs_iter < VTK_Contents.totalBoxElements; cgs_iter++) {
            //Iterations of the Conjugate Gradient Solver here
            vectorDot_GPU << <numBlocks1D, threadsPerBlock1D, 0, stream[0] >> > (d_rk, d_rk, d_r_norm_old, d_SolverConfig); //r_k dot r_k
            MatrixMul_Omnidirectional_GPU << <numBlocks3D, threadsPerBlock3D, 0, stream[1] >> > (d_temp, d_pk, d_RHS, d_SolverConfig, d_gridCoeffs, d_Wxp, d_Wxm, d_Wyp, d_Wym, d_Wzp, d_Wzm); //temp=A*p_k
            cudaDeviceSynchronize();
            vectorDot_GPU << <numBlocks1D, threadsPerBlock1D >> > (d_pk, d_temp, d_temp_scal, d_SolverConfig); //temp_scal = p_k dot temp
            cudaDeviceSynchronize();
            divide << <1, 1, 0, stream[0] >> > (d_r_norm_old, d_temp_scal, d_alpha);//alpha = (rk dot rk) / (pk dot A*pk)

            //Implicit residual update
            scalarVectorMult_GPU << <numBlocks1D, threadsPerBlock1D, 0, stream[1] >> > (d_alpha, d_temp, d_temp, d_SolverConfig); //temp=alphak*temp
            subtractVectors_GPU << <numBlocks1D, threadsPerBlock1D, 0, stream[2] >> > (d_rk, d_temp, d_rkp1, d_SolverConfig); //r_k+1=rk-temp (i.e. rk-A*temp)

            cudaDeviceSynchronize();
            scalarVectorMult_GPU << <numBlocks1D, threadsPerBlock1D, 0, stream[0] >> > (d_alpha, d_pk, d_temp, d_SolverConfig); //temp = alphak*pk
            cudaDeviceSynchronize();
            addVectors_GPU << <numBlocks1D, threadsPerBlock1D, 0, stream[0] >> > (d_PressureField, d_temp, d_PressureField, d_SolverConfig); //xk+1=xk+alphak*pk
            cudaDeviceSynchronize();
            //printVector1_GPU << <1, 1 >> > (d_PressureField); cudaDeviceSynchronize();
            //printVector2_GPU << <1, 1 >> > (d_PressureField);

            //Explicit residual update
                //MatrixMul_Omnidirectional_GPU << <numBlocks3D, threadsPerBlock3D >> > (d_temp, d_PressureField, d_RHS, d_SolverConfig, d_gridCoeffs, d_Wxp, d_Wxm, d_Wyp, d_Wym, d_Wzp, d_Wzm); //temp=A*x_k+1
                //cudaDeviceSynchronize();
                //subtractVectors_GPU << <numBlocks1D, threadsPerBlock1D >> > (d_RHS, d_temp, d_rkp1, d_SolverConfig); //r_k+1=b-A*xk+1
                //cudaDeviceSynchronize();

            cudaMemcpyAsync(d_rk, d_rkp1, boxArraySize, cudaMemcpyDeviceToDevice, stream[0]); //rk=rk+1
            vectorDot_GPU << <numBlocks1D, threadsPerBlock1D, 0, stream[1] >> > (d_rkp1, d_rkp1, d_r_norm, d_SolverConfig); //r_k+1 dot r_k+1
            cudaDeviceSynchronize();

            varfloat r_norm; cudaMemcpy(&r_norm, d_r_norm, sizeof(varfloat), cudaMemcpyDeviceToHost);
            cudaDeviceSynchronize();
            r_norm = sqrt(r_norm);

            if (cgs_iter % 10 == 0) {
                printf("Pressure Solver Iteration=%d; CG Iteration=%d; RelRes=%0.2e;  AbsRes=%0.2e\n", pressureIterations + 1, cgs_iter, r_norm / r_norm_init, r_norm);
            }

            if ((r_norm / r_norm_init > SolverConfig.solverToleranceRel) && (r_norm > SolverConfig.solverToleranceAbs)) {
                //Only continues if not yet within tolerance
                divide << <1, 1 >> > (d_r_norm, d_r_norm_old, d_beta);//beta = (rk+1 dot rk+1) / (rk dot rk)
                cudaDeviceSynchronize();
                scalarVectorMult_GPU << <numBlocks1D, threadsPerBlock1D >> > (d_beta, d_pk, d_temp, d_SolverConfig); //temp=beta*pk
                cudaDeviceSynchronize();
                addVectors_GPU << <numBlocks1D, threadsPerBlock1D >> > (d_temp, d_rkp1, d_pk, d_SolverConfig); //pk+1=rk+1 + beta*pk 
                cudaDeviceSynchronize();
            }
            else {
                printf("Pressure Solver Iteration=%d; CG Iteration=%d; RelRes=%0.2e;  AbsRes=%0.2e [Converged]\n", pressureIterations + 1, cgs_iter, r_norm / r_norm_init, r_norm);
                break;
            }
        }

        if (pressureIterations > 0) {
            //For iterations 2 and onwards, attempts to use over-relaxation to make convergence faster
            subtractVectors_GPU << <numBlocks1D, threadsPerBlock1D >> > (d_PressureField, d_PressureFieldOld, d_temp, d_SolverConfig); //temp=P-Pold
            cudaDeviceSynchronize();

            //Also computes the norm of the difference to assess convergence
            vectorDot_GPU << <numBlocks1D, threadsPerBlock1D, 0, stream[1] >> > (d_temp, d_temp, d_temp_scal, d_SolverConfig); //temp_scal = (P-Pold) dot (P-Pold)
            cudaMemcpy(&delta_p_norm, d_temp_scal, sizeof(varfloat), cudaMemcpyDeviceToHost);
            delta_p_norm = sqrt(delta_p_norm);

            //Also gets the norm of the pressure field to assess convergence
            vectorDot_GPU << <numBlocks1D, threadsPerBlock1D, 0, stream[1] >> > (d_PressureField, d_PressureField, d_temp_scal, d_SolverConfig); //temp_scal = P dot P
            cudaMemcpy(&p_norm, d_temp_scal, sizeof(varfloat), cudaMemcpyDeviceToHost);
            p_norm = sqrt(p_norm);

            scalarVectorMult_GPU << <numBlocks1D, threadsPerBlock1D >> > (d_over_relax, d_temp, d_temp, d_SolverConfig); //temp=over-relax * temp
            cudaDeviceSynchronize();
            addVectors_GPU << <numBlocks1D, threadsPerBlock1D >> > (d_PressureFieldOld, d_temp, d_PressureField, d_SolverConfig); //P=Pold+overRelax*temp
            cudaDeviceSynchronize();
        }

        cudaDeviceSynchronize();
        cudaMemcpyAsync(d_PressureFieldOld, d_PressureField, boxArraySize, cudaMemcpyDeviceToDevice, stream[0]); //Copies pressure field to pressure_old
        vectorDot_GPU << <numBlocks1D, threadsPerBlock1D, 0, stream[1] >> > (d_PressureField, d_PressureField, d_temp_scal, d_SolverConfig); //Takes the norm of the Pressure Field x_k for output
        cudaDeviceSynchronize();
        varfloat x_norm; cudaMemcpy(&x_norm, d_temp_scal, sizeof(varfloat), cudaMemcpyDeviceToHost);
        x_norm = sqrt(x_norm);
        varfloat relativeResidualNorm = abs(r_norm_init / first_r_norm_init);
        printf("Relative pressure residual norm:%f)\n", relativeResidualNorm);

        varfloat relativePressureNorm = abs(delta_p_norm / p_norm);
        printf("Relative pressure difference norm:%f)\n", relativePressureNorm);

        //checks if change is less than tolerance
        if (relativeResidualNorm < SolverConfig.PressureSolverToleranceRel) {
            printf("Pressure field norm change below tolerance. Exiting.\n");
            break;
        }

        //Provides user with timing for this round
        ClockToc("CG Solver Iteration Time:");


        //Prints out info about iterations to csv file
        toc2 = clock() - tic2;
        timeTask = ((varfloat)toc2) / CLOCKS_PER_SEC; // in seconds
        string stringOut = ""; stringOut.append(to_string(pressureIterations + 1)); stringOut.append(",");
        stringOut.append(to_string(relativeResidualNorm)); stringOut.append(",");
        stringOut.append(to_string(relativePressureNorm)); stringOut.append(",");
        stringOut.append(to_string(timeTask)); stringOut.append("\n");
        csvOutput->write(stringOut.c_str(), stringOut.size() * sizeof(char));

        //Saves a checkpoint iteration
        if ((SolverConfig.CheckpointIterations > 0) && ((pressureIterations + 1) % SolverConfig.CheckpointIterations == 0)) {
            printf("\n~~~Saving checkpoint iteration %d...~~\n", pressureIterations + 1);
            cudaMemcpy(PressureField, d_PressureField, boxArraySize, cudaMemcpyDeviceToHost);
            SaveVTK(OutputFileName, PressureField, SolverConfig, VTK_Contents);
            printf("\nSaved, Successfully!\n");
        }
    }

    //Extracts 3D array from GPU Memory
    cudaMemcpy(PressureField, d_PressureField, boxArraySize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    //Frees memory 
    printf("==========================================================\n");
    cudaFree(d_PressureField); cudaFree(d_RHS); cudaFree(d_rk); cudaFree(d_rkp1); cudaFree(d_pk); cudaFree(d_temp);
    cudaFree(d_Wxp); cudaFree(d_Wxm); cudaFree(d_Wyp); cudaFree(d_Wym); cudaFree(d_Wzp); cudaFree(d_Wzm);
    cudaFree(d_SourceX); cudaFree(d_SourceY); cudaFree(d_SourceZ);
    cudaFree(d_gridCoeffs); cudaFree(d_SolverConfig);
    cudaFree(d_beta); cudaFree(d_alpha); cudaFree(d_r_norm); cudaFree(d_r_norm_old); cudaFree(d_temp_scal);

    //Destroy parallel streams
    for (int i = 0; i < nStreams; ++i) {
        cudaStreamDestroy(stream[i]);
    }
}



#pragma endregion



int main() {
    //Assumes single precision
    SolverParameters<varfloat> SolverConfig;
    //Reads input file to configure cameras
    printf("==============Reading Input Parameter File...==============\n");
    string fileName = "Arguments.conf";
    if (!ParseInputParameterFile(fileName, SolverConfig)) {
        //Aborts; file format is wrong.
        printf("Error reading input file! Aborting.\n");
        return 0;
    }
    
    if (!ParseInputParameterFile(fileName, SolverConfig)) {
        //Aborts; file format is wrong.
        printf("Error reading input file! Aborting.\n");
        return 0;
    }


    //Figures out how many VTK files to render by looking into the filename convention
    vector <string> BoxDataFileList;
    vector <int> BoxDataFileNumber;
    path pathObj(SolverConfig.BoxInputFile);
    string BoxPath = pathObj.parent_path().string();
    string BoxFileName = pathObj.filename().string();

    //First let's see if the string <frame> is even used here
    size_t pos = BoxFileName.find("<frame>");
    if (pos == std::string::npos) {
        //String <frame> not used, so must be a single file.
        printf("No pattern <frame> found, so using the full file name below as the VTK file:\n %s\n", SolverConfig.BoxInputFile.c_str());
        BoxDataFileList.push_back(SolverConfig.BoxInputFile);
        BoxDataFileNumber.push_back(0);
    }
    else {
        //String <frame> not used, so let's find all instances of files using that string
        string BeforeFrame = BoxFileName.substr(0, pos);
        string AfterFrame = BoxFileName.substr(pos + 7, BoxFileName.length() - (pos + 7));

        replace(BoxFileName, "<frame>", "\\d+\\");
        regex regex_pattern(BoxFileName);

        printf("Searching Directory %s \n for files of the pattern %s...\n", BoxPath.c_str(), BoxFileName.c_str());
        for (const auto& entry : directory_iterator(BoxPath)) { //Goes through the directory and finds the files
            string ThisFileName = entry.path().filename().string();
            printf("Looking at file %s...\n", ThisFileName.c_str());
            if (is_regular_file(entry) && regex_match(ThisFileName, regex_pattern)) {
                string ss = ThisFileName;
                eraseSubStr(ss, BeforeFrame);
                eraseSubStr(ss, AfterFrame);

                int number = std::stoi(ss);
                printf("-- File matches expression. File Number = %d.\n", number);
                BoxDataFileList.push_back(entry.path().string());
                BoxDataFileNumber.push_back(number);
            }
        }
    }


    for (int vtkIdx = 0; vtkIdx < BoxDataFileList.size(); vtkIdx++) {
        int ThisFileNumber = BoxDataFileNumber[vtkIdx];
        string ThisBoxFile = BoxDataFileList[vtkIdx];

        printf("\n\n\n ========Initializing Solver for file '%s'========\n", ThisBoxFile.c_str());

        ClockTic();
        BoxContents<varfloat> VTK_Contents;
        //===========Reads the box contents file===========
        if (!ReadVTK(ThisBoxFile, VTK_Contents)) {
            //Aborts; file format is wrong.
            printf("Error reading box data file! Aborting.\n");
            return 0;
        }
        SolverConfig.BoxGridPoints = VTK_Contents.BoxGridSize; // Ensure the variables have the same value
        SolverConfig.GridDelta = VTK_Contents.GridDelta;
        SolverConfig.totalBoxElements = VTK_Contents.totalBoxElements;

        //Configure output file for VolumeSum metric results
        ofstream csvOutput("Residual_RelativeNorm.csv", std::ios::out);
        if (!csvOutput) {
            printf("Error writing output CSV file!\n");
            return false;
        }
        string stringOut = "Iteration, Residual Norm, Pressure Difference Norm, Time [s]\n"; //header
        csvOutput.write(stringOut.c_str(), stringOut.size() * sizeof(char));

        ClockToc("File Read Time:");

        //===========Computes the weight coefficients===========
        printf("========Initializing Grid Weight Coefficients========\n");
        ClockTic();

        gridCoefficients<varfloat> gridCoeffs;
        if (VTK_Contents.BoxGridSize.z == 1) {
            //Coefficients are 2D
            gridCoefficients2D(VTK_Contents, gridCoeffs);
        }
        else {
            //Coefficients are 3D
            gridCoefficients3D(VTK_Contents, gridCoeffs, 1e7);
        }

        ClockToc("Grid Coefficient Computation Time:");

        //~~~~~~~~~~~~===========LOADS DATA INTO GPU AND FIRES UP THE CONJUGATE GRADIENT SOLVER===========~~~~~~~~~~~~~~~~~~~~~
        printf("========Starting up the solver...========\n");

        ClockTic();
        // Allocate CPU memory for the result
        varfloat* PressureField;
        long long boxArraySize = sizeof(varfloat) * VTK_Contents.totalBoxElements;
        PressureField = (varfloat*)malloc(boxArraySize);
        FillBox(PressureField, ZEROS, SolverConfig); // Initializes with zeros

        //Starts solving the pressure equations on a loop
        varfloat* RHS; RHS = (varfloat*)malloc(boxArraySize);
        FillBox(RHS, ZEROS, SolverConfig); // Initializes with zeros
        varfloat CurrentResultNorm = 0.0;

        // Precompute the weights
        BoxWeights< varfloat> Weights;
        Weights.Wxp = (varfloat*)malloc(boxArraySize); Weights.Wxm = (varfloat*)malloc(boxArraySize);
        Weights.Wyp = (varfloat*)malloc(boxArraySize); Weights.Wym = (varfloat*)malloc(boxArraySize);
        Weights.Wzp = (varfloat*)malloc(boxArraySize); Weights.Wzm = (varfloat*)malloc(boxArraySize);
        Precompute_Weights_Iteration_nPlusOne(Weights, SolverConfig, VTK_Contents, gridCoeffs);

        ClockToc("Weight Precomputation Time:");

        string OutputFileName = SolverConfig.BoxOutputFile;
        replace(OutputFileName, "<frame>", to_string(ThisFileNumber));

        ClockTic2();
        //Solves the equations with the solver
        ConjugateGradientSolver_GPU(PressureField, RHS, Weights, SolverConfig, VTK_Contents, gridCoeffs, OutputFileName, &csvOutput);
        ClockToc2("Total Time for Pressure Solver:");

        //===========Saves the result for the current file===========
        SaveVTK(OutputFileName, PressureField, SolverConfig, VTK_Contents);
        printf("\nSaved Output File Successfully!\n");

        free(PressureField); free(RHS);

        //Closes CSV output
        csvOutput.close();
    }

    return 0;
}

