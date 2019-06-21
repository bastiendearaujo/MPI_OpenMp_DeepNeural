#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "preprocessing.h"
#include "utils.h"

int currentRaw;
int currentCol;
int sizeOfTableProtocol;
int sizeOfTableService;
int sizeOfTableFlag;
int sizeOfTableOutput;
int nbRawMatrix;
int nbColMatrix;

char ** protocolTable;
char ** serviceTable;
char ** flagTable;
int * nbOutTable;

void displayErrorFind(char ** outTable, int * nbErrorFind){
    for (int i = 0; i < sizeOfTableOutput; ++i){
        printf("%s : %d / %d | difference : %d \n", outTable[i] ,nbErrorFind[i], nbOutTable[i], nbErrorFind[i]-nbOutTable[i]);
    }
}

char* getfield(char* line, int num){
    char* tok;
    for (tok = strtok(line, ",");
            tok && *tok;
            tok = strtok(NULL, ",\n"))
    {
        if (!--num)
            return tok;
    }
    return NULL;
}

void readFile(char * nameFile, char ** outTable){
    nbRawMatrix = 0;
    FILE* stream = fopen(nameFile, "r");
    sizeOfTableProtocol = 0;
    sizeOfTableService = 0;
    sizeOfTableFlag = 0;
    sizeOfTableOutput = 0;
    int i;
    char line[1024];
    
    protocolTable = malloc(sizeof(char*)*100);
    #pragma omp for schedule(static) private(i)
    for (i = 0; i < 100; i++)
        protocolTable[i] = malloc(100 * sizeof(char));

    serviceTable = malloc(sizeof(char*)*100);
    #pragma omp for schedule(static) private(i)
    for (i = 0; i < 100; i++)
        serviceTable[i] = malloc(100 * sizeof(char));

    flagTable = malloc(sizeof(char*)*100);
    #pragma omp for schedule(static) private(i)
    for (i = 0; i < 100; i++)
        flagTable[i] = malloc(100 * sizeof(char));

    


    while (fgets(line, 1024, stream)){
        char* tmp = strdup(line);
        char* protocol = getfield(tmp, 2);

        tmp = strdup(line);
        char* service = getfield(tmp, 3);

        tmp = strdup(line);
        char* flag = getfield(tmp, 4);

        tmp = strdup(line);
        char* out = getfield(tmp, 42);
        
        if(!isPresentTab(protocolTable, sizeOfTableProtocol, protocol)){
            strcpy(protocolTable[sizeOfTableProtocol], protocol);
            sizeOfTableProtocol++;
        }  

        if(!isPresentTab(serviceTable, sizeOfTableService, service)){
            strcpy(serviceTable[sizeOfTableService], service);
            sizeOfTableService++;
        }  

        if(!isPresentTab(flagTable, sizeOfTableFlag, flag)){
            strcpy(flagTable[sizeOfTableFlag], flag);
            sizeOfTableFlag++;
        }

        if(!isPresentTab(outTable, sizeOfTableOutput, out)){
            strcpy(outTable[sizeOfTableOutput], out);
            sizeOfTableOutput++;
        }

        free(tmp);
        nbRawMatrix ++;
    }
    fclose(stream);
    
    #ifdef ALLINF
        printTab(protocolTable, sizeOfTableProtocol);
        printf("(%d)\n", sizeOfTableProtocol);

        printTab(serviceTable, sizeOfTableService);
        printf("(%d)\n", sizeOfTableService);

        printTab(flagTable, sizeOfTableFlag);
        printf("(%d)\n", sizeOfTableFlag);

        printTab(outTable, sizeOfTableOutput);
        printf("(%d)\n", sizeOfTableOutput);
    #endif

    nbOutTable = malloc(sizeof(int)*sizeOfTableOutput);
    for (int i = 0; i < sizeOfTableOutput; ++i){
        nbOutTable[i] = 0;
    }

    nbColMatrix = sizeOfTableProtocol + sizeOfTableService + sizeOfTableFlag + 38;
}

void makeVector(int currentColinRaw, char * field, double * matrix, int * vectorOutput, char ** outTable){
    int i;
    if (currentColinRaw == 1){
        for(i = 0; i < sizeOfTableProtocol; i++){
            if(!strcmp(field, protocolTable[i])){
                matrix[currentRaw*nbColMatrix+currentCol] = 1;
            }else{
                matrix[currentRaw*nbColMatrix+currentCol] = 0;
            }
            currentCol++;
        }
    }else if (currentColinRaw == 2){
        for(i = 0; i < sizeOfTableService; i++){
            if(!strcmp(field, serviceTable[i])){
                matrix[currentRaw*nbColMatrix+currentCol] = 1;
            }else{
                matrix[currentRaw*nbColMatrix+currentCol] = 0;
            }
            currentCol++;
        }
    }else if (currentColinRaw == 3){
        for(i = 0; i < sizeOfTableFlag; i++){
            if(!strcmp(field, flagTable[i])){ // Ŝame element
                matrix[currentRaw*nbColMatrix+currentCol] = 1;
            }else{
                matrix[currentRaw*nbColMatrix+currentCol] = 0;
            }
            currentCol++;
        }
    }else if(currentColinRaw == 41){
        for(i = 0; i < sizeOfTableOutput; i++){
            if(!strcmp(field, outTable[i])){
                vectorOutput[currentRaw] = i;
                nbOutTable[i]++;
            }else{

            }
        }
    }else {
        matrix[currentRaw*nbColMatrix+currentCol] = tanh(atof(field));
        currentCol++;
    }

}

void makeMatrix(char * nameFile, double * matrix, int * vectorOutput, char ** outTable){
    currentRaw = 0;
    #ifdef ALLINF
        printf("\nMatrix in creation\n");
    #endif

    char line[1024];
    int i;
    FILE* stream = fopen(nameFile, "r");
    char* field;
    while (fgets(line, 1024, stream)){
        currentCol=0;
        for (i = 0; i <= NBCOL; i++){
            char* tmp = strdup(line);
            field = getfield(tmp, i+1);
            makeVector(i, field, matrix, vectorOutput, outTable);
        }
        currentRaw ++; 
    }

    #ifdef ALLINF
        printf("Ligne 1 : \n");
        displayMatrix(matrix, 1, nbColMatrix);
        printf("Matrix created.\n");
    #endif
}

int getNbColMatrix(){
    return nbColMatrix;
}

int getNbRawMatrix(){
    return nbRawMatrix;
}

int getSizeOfTableOutput(){
    return sizeOfTableOutput;
}