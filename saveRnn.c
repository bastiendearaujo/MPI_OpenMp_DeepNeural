#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <assert.h>
#include <time.h>

#include "utils.h"
// #include "preprocessing.h"

#define NBCOL 41
#define NBLAYER 5
#define NBHD 120

int layerSize[NBLAYER];

typedef struct layer LAYER;
struct layer {
  int typeLayer;
  int nbNodes;
  double* bias;
  double* value;
  double* value_prev;
  double* error;
  double* error_prev;
  double* weight;
};

LAYER * tabLayer[NBLAYER];

int sizeOfTableProtocol;
int sizeOfTableService;
int sizeOfTableFlag;
int sizeOfTableOutput;
int nbRawMatrix;
int nbColMatrix;
char ** protocolTable;
char ** serviceTable;
char ** flagTable;
char ** outTable;
double * matrix;
int * vectorOutput;
int * nbOutTable;
int * nbErrorFind;
int currentRaw;
int currentCol;

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

void readFile(char * nameFile){
    FILE* stream = fopen(nameFile, "r");
    sizeOfTableProtocol = 0;
    sizeOfTableService = 0;
    sizeOfTableFlag = 0;
    sizeOfTableOutput = 0;

    char line[1024];
    
    protocolTable = malloc(sizeof(char*)*100);
    for (int i = 0; i < 100; i++)
        protocolTable[i] = malloc(100 * sizeof(char));

    serviceTable = malloc(sizeof(char*)*100);
    for (int i = 0; i < 100; i++)
        serviceTable[i] = malloc(100 * sizeof(char));

    flagTable = malloc(sizeof(char*)*100);
    for (int i = 0; i < 100; i++)
        flagTable[i] = malloc(100 * sizeof(char));

    outTable = malloc(sizeof(char*)*100);
    for (int i = 0; i < 100; i++)
        outTable[i] = malloc(100 * sizeof(char));


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

void makeVector(int currentColinRaw, char * field){
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

void makeMatrix(char * nameFile){
    currentRaw = 0;
    matrix = malloc(sizeof(double)*nbColMatrix*nbRawMatrix);
    vectorOutput = malloc(sizeof(int)*nbRawMatrix);

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
            makeVector(i, field);
        }
        currentRaw ++; 
    }

    #ifdef ALLINF
        printf("Ligne 1 : \n");
        displayMatrix(matrix, 1, nbColMatrix);
        printf("Matrix created.\n");
    #endif
    // displayVector(vectorOutput, nbRawMatrix);
}

void preprocessing(char * nameFile){
    readFile(nameFile);
    makeMatrix(nameFile);
}

void init_layer(LAYER * layer, int raw, int currentLayer){
    layer->typeLayer = currentLayer;

    int i;
    // Init Input Layer
    if (currentLayer == 0){
        layer->nbNodes = nbColMatrix;
        // Init vector input layer
        layer->value = malloc(sizeof(double)*nbColMatrix);
        for (i=0; i<nbColMatrix; i++) {
            layer->value[i] = matrix[nbColMatrix*raw+i];
        }
        // Init weight 
        layer->weight = malloc(sizeof(double)*nbColMatrix*layerSize[currentLayer+1]);
        for (i = 0; i < nbColMatrix*layerSize[currentLayer+1]; ++i){
            layer->weight[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
        }
    }else if (currentLayer == NBLAYER-1){ // Init Output Layer
        layer->nbNodes = layerSize[currentLayer];
        layer->value = malloc(sizeof(double)*layerSize[currentLayer]);
    }else{  // Init Hidden Layer
        layer->nbNodes = layerSize[currentLayer];
        layer->value = malloc(sizeof(double)*layerSize[currentLayer]);
        for (i=0; i<layerSize[currentLayer]; i++) {
            layer->value[i] = 0.0;
        }

        layer->weight = malloc(sizeof(double) * layerSize[currentLayer+1] * layerSize[currentLayer]);
        for (i = 0; i < layerSize[currentLayer+1]*layerSize[currentLayer]; ++i){
            layer->weight[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
        }
    }

    layer->bias = malloc(sizeof(double)*layerSize[currentLayer]);
    for (i = 0; i < layerSize[currentLayer]; ++i){
        layer->bias[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
    }
    layer->value_prev = malloc(sizeof(double)*layerSize[currentLayer]);
    for (i=0; i<layerSize[currentLayer]; i++) {
        layer->value_prev[i] = 0.0;
    }
    layer->error = malloc(sizeof(double)*layerSize[currentLayer]);
    for (i=0; i<layerSize[currentLayer]; i++) {
        layer->error[i] = 0.0;
    }
    layer->error_prev = malloc(sizeof(double)*layerSize[currentLayer]);
    for (i=0; i<layerSize[currentLayer]; i++) {
        layer->error_prev[i] = 0.0;
    }

}

void rnnsetstart(LAYER * tabLayer[]){
    int i, k;
    for (i = 0; i < NBLAYER; ++i){
        if (tabLayer[i]->typeLayer == NBLAYER-1){
            for (k = 0; k < tabLayer[i]->nbNodes; ++k){
                tabLayer[i]->value_prev[k] = sigmoid(tabLayer[i]->value[k]);
            }
        }else{
            for (k = 0; k < tabLayer[i]->nbNodes; ++k){
                tabLayer[i]->value_prev[k] = tabLayer[i]->value[k];
            }
        }
    }
}

void rnnset(LAYER * tabLayer[]){
    int i,j;

    for (i = 0; i < NBLAYER; ++i){
        // If not input
        if(tabLayer[i]->typeLayer != 0){
            for (j = 0; j < tabLayer[i]->nbNodes; ++j){
                tabLayer[i]->value[j] = tabLayer[i]->bias[j];
            }
            matrixTimesMatrixTan(tabLayer[i-1]->value, tabLayer[i-1]->value_prev, tabLayer[i-1]->weight, tabLayer[i]->value, tabLayer[i-1]->bias, tabLayer[i-1]->nbNodes, tabLayer[i]->nbNodes);            
            // softmax(tabLayer[i]->value, tabLayer[i]->nbNodes);
        }
    }
}

void rnnlearn(LAYER * tabLayer[], double * out, double learningrate){
    int i,j,k;
    double * derivative;
    double * wchange;

    for(i = 0; i < NBLAYER; i++){
        for (j = 0; j < tabLayer[i]->nbNodes; ++j){
            tabLayer[i]->error[j] = 0.0;
        }

        if (tabLayer[i]->typeLayer == NBLAYER-1){
            // Substraction vector with OUT and final out
            vectorSubstraction(tabLayer[i]->value, out, tabLayer[i]->error, tabLayer[i]->nbNodes);
        }
    }

    for(i = NBLAYER-2; i>= 0; i--){
        matrixTimesMatrix(tabLayer[i+1]->error, tabLayer[i]->weight, tabLayer[i]->error, tabLayer[i+1]->nbNodes, tabLayer[i]->nbNodes);
    }

    for(i = NBLAYER-2; i>= 0; i--){
        derivative = malloc(sizeof(double)*tabLayer[i+1]->nbNodes);
        wchange = malloc(sizeof(double)*tabLayer[i+1]->nbNodes*tabLayer[i]->nbNodes);

        for (j = 0; j < tabLayer[i+1]->nbNodes; ++j){
            derivative[j] = 0.0;
            for (k = 0; k < tabLayer[i]->nbNodes; ++k){
                wchange[k*tabLayer[i+1]->nbNodes+j] = 0.0;
            }
        }

        if(i == NBLAYER-2){

        // for (j = 0; j < tabLayer[i]->nbNodes; ++j){
        //     for (k = 0; k < tabLayer[i+1]->nbNodes; ++k){
        //         tmp = tabLayer[i+1]->error[k] * learningrate * tabLayer[i]->value[j];
        //         if(i != NBLAYER-2){
        //             tmp *= 1 - (tabLayer[i+1]->value[j] * tabLayer[i+1]->value[j]);
        //         }
        //         tabLayer[i]->weight[j*tabLayer[i+1]->nbNodes+k] -= tmp;

        //         // Normalize weight
        //         if(tabLayer[i]->weight[j*tabLayer[i+1]->nbNodes+k]>normalize){
        //             tabLayer[i]->weight[j*tabLayer[i+1]->nbNodes+k]=normalize;
        //         }else if(tabLayer[i]->weight[j*tabLayer[i+1]->nbNodes+k]<(-normalize)){
        //             tabLayer[i]->weight[j*tabLayer[i+1]->nbNodes+k] = (-normalize);
        //         }
        //     }
        // }

        // vectorMinusVector(tabLayer[i+1]->bias, tabLayer[i+1]->error, tabLayer[i]->nbNodes);
        // normalisation(tabLayer[i+1]->bias, 5, tabLayer[i]->nbNodes);

            vectorTimesDouble(tabLayer[i+1]->error, derivative, learningrate, tabLayer[i+1]->nbNodes, 0);
            
            for (j = 0; j < tabLayer[i+1]->nbNodes; ++j){
                for (k = 0; k < tabLayer[i]->nbNodes; ++k){
                    wchange[k*tabLayer[i+1]->nbNodes+j] = derivative[j];
                }
            }
            
            // wchange *= value i+1
            matrixScalaire(wchange, tabLayer[i]->value, tabLayer[i+1]->nbNodes, tabLayer[i]->nbNodes);
            // weight i -= wchange;
            matrixMinusMatrix(tabLayer[i]->weight, wchange, tabLayer[i]->nbNodes, tabLayer[i+1]->nbNodes);
            // Normalisation de weight -5 et 5
            normalisation(tabLayer[i]->weight, 5, tabLayer[i]->nbNodes*tabLayer[i+1]->nbNodes);
            // bias i+1 -= derivative
            vectorMinusVector(tabLayer[i]->bias, derivative, tabLayer[i]->nbNodes);
            // Normalisation de bias -5 et 5
            normalisation(tabLayer[i]->bias, 5, tabLayer[i]->nbNodes);

        }
        else{ // If not Output Layer
            // derivative = 1.0 - value i+1 ^ 2
            computeDerivative(derivative, tabLayer[i+1]->value, tabLayer[i+1]->nbNodes);

            // derivative *= error i + 1 * learningrate
            vectorTimesDouble(tabLayer[i+1]->error, derivative, learningrate, tabLayer[i+1]->nbNodes, 1);
            
            for (j = 0; j < tabLayer[i+1]->nbNodes; ++j){
                for (k = 0; k < tabLayer[i]->nbNodes; ++k){
                    wchange[k*tabLayer[i+1]->nbNodes+j] = derivative[j];
                }
            }

            // wchange *= value i+1
            matrixScalaire(wchange, tabLayer[i]->value, tabLayer[i+1]->nbNodes, tabLayer[i]->nbNodes);
            // weight i -= wchange
            matrixMinusMatrix(tabLayer[i]->weight, wchange, tabLayer[i]->nbNodes, tabLayer[i+1]->nbNodes);
            // bias i -= derivative
            vectorMinusVector(tabLayer[i]->bias, derivative, tabLayer[i]->nbNodes);
        }

        free(wchange);
        free(derivative);
    }

    // Update error_prev:
    for(i = 0; i < NBLAYER; i++){
        for (j = 0; j < tabLayer[i]->nbNodes; ++j){
            tabLayer[i]->error_prev[j] = tabLayer[i]->error[j];
        }
    }
}

double geterror(LAYER* tabLayer){
    double res =0.0;
    for (int i = 0; i < tabLayer->nbNodes; ++i){ 
        if(tabLayer->error[i]<0){
            res -= tabLayer->error[i];
        }else{
            res += tabLayer->error[i];
        }
    }
    return res/(tabLayer->nbNodes);
}

void displayResult(LAYER * layer){
    for (int i = 0; i < layer->nbNodes; ++i){ 
        printf("Result [%s] = %f\n",outTable[i], layer->value[i]);
    }
}

void fillOutc(double * out, int raw){
    for (int i = 0; i < sizeOfTableOutput; ++i){
        if (i == vectorOutput[raw]){
            out[i] = 1.0;
        }else{
            out[i] = 0.0;
        }
    }
}

void initLayerSize(){
    int i;
    layerSize[0] = nbColMatrix; 
    for (i = 1; i < NBLAYER-1; ++i){
        layerSize[i] = NBHD;
    }
    layerSize[NBLAYER-1] = sizeOfTableOutput;
}

void wichError(LAYER * layer){
    double max = 0.0;
    char * res;
    ((void) res);
    int indiceMax =0;
    for (int i = 0; i < layer->nbNodes; ++i){ 
        if (layer->value[i]>max){
            max = layer->value[i];
            res = outTable[i];
            indiceMax = i;
        }
    }

    nbErrorFind[indiceMax]++; 
    #ifdef ALLINF      
        printf("Type erreur : %s   | Sureté : %f \n", res, max);
    #endif
}

void freeLayer(LAYER * layer){
    free(layer->value);
    free(layer->weight);
    free(layer->error);
    free(layer->value_prev);
    free(layer->error_prev);
    free(layer->bias);
}

void ajustError(LAYER * layer){
    int i;
    double sum = 0.0;
    for (i = 0; i < layer->nbNodes; i++){ 
        sum += layer->value[i];
    }

    for (i = 0; i < layer->nbNodes; i++){ 
        layer->value[i] /= sum;
    }
}

void learnKDD(){
    nbRawMatrix = 0;
    preprocessing("./exampleFiles/KDDTest+.txt");
    nbErrorFind = malloc(sizeof(int)*sizeOfTableOutput);
    for (int i = 0; i < sizeOfTableOutput; ++i){
        nbErrorFind[i] = 0;
    }

    initLayerSize();

    for (int i = 0; i < NBLAYER; ++i){
        tabLayer[i] = (LAYER*)malloc(sizeof(LAYER));
    }
    for (int i = 0; i < NBLAYER; ++i){
        init_layer(tabLayer[i], 0, i);
    }

    // RUN
    double error = 1;
    double * outc = malloc(sizeof(double)*sizeOfTableOutput);
    double learningrate = 0.03;
    int i, j, sum=0;

    for (i = 0; i < nbRawMatrix; ++i){ //nbRawMatrix
        error = 1;
        j = 0;
        init_layer(tabLayer[0], i, 0);

        fillOutc(outc, i);

        while (error > 0.01 && j<1000) {
            rnnsetstart(tabLayer);
            rnnset(tabLayer);
            rnnlearn(tabLayer,outc,learningrate);
            error = geterror(tabLayer[NBLAYER-1]);
            j++;
        }

        ajustError(tabLayer[NBLAYER-1]);
        #ifdef ALLINF      
            printf("\nLigne : %d | trouvé en : %d iteration  | ", i+1, j);
        #endif

        wichError(tabLayer[NBLAYER-1]);
        #ifdef ALLINF      
            displayResult(tabLayer[NBLAYER-1]);
        #endif


        freeLayer(tabLayer[0]);

        sum+=j;

    }
    
    free(outc);

    printf("\nMoyenne iteration par ligne  : %d \n", sum / nbRawMatrix);

    printf("\nError Find in learnKDD : \n");
    for (int i = 0; i < sizeOfTableOutput; ++i){
        printf("%s : %d / %d \n", outTable[i] ,nbErrorFind[i], nbOutTable[i]);
    }
}

void testKDD(){
    nbRawMatrix = 0;
    preprocessing("./exampleFiles/KDDTest+.txt");

    nbErrorFind = malloc(sizeof(int)*sizeOfTableOutput);
    for (int i = 0; i < sizeOfTableOutput; ++i){
        nbErrorFind[i] = 0;
    }

    initLayerSize();
    // RUN
    double error =1;
    double * outc = malloc(sizeof(double)*sizeOfTableOutput);
    double learningrate = 0.1;
    int i, j;
    for (i = 0; i < nbRawMatrix; ++i){ //nbRawMatrix
        error = 1;
        j = 0;
        init_layer(tabLayer[0], i, 0);
        fillOutc(outc, i);
        
        rnnsetstart(tabLayer);
        rnnset(tabLayer);
        rnnlearn(tabLayer,outc,learningrate);
        error = geterror(tabLayer[NBLAYER-1]);
        j++;

        ajustError(tabLayer[NBLAYER-1]);

        #ifdef ALLINF      
            printf("\nLigne : %d | trouvé en : %d iteration  | ", i+1, j);
        #endif

        wichError(tabLayer[NBLAYER-1]);
        #ifdef ALLINF      
            displayResult(tabLayer[NBLAYER-1]);
        #endif


        freeLayer(tabLayer[0]);
    }

    printf("\nError Find in TestKDD : \n");
    for (int i = 0; i < sizeOfTableOutput; ++i){
        printf("%s : %d / %d | difference : %d \n", outTable[i] ,nbErrorFind[i], nbOutTable[i], nbErrorFind[i]-nbOutTable[i]);
    }
}

int main(int argc, char ** argv){
    double tbeg, elapsedTime;
    int nb_procs, rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nb_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0){

        printf("Begin Learn\n");
        tbeg = MPI_Wtime();
        learnKDD();
        elapsedTime = MPI_Wtime() - tbeg;
        printf("Time to execute learnKDD : %f\n", elapsedTime);

        printf("End Learn, begin test\n");
        tbeg = MPI_Wtime();    
        testKDD();
        elapsedTime = MPI_Wtime() - tbeg;
        printf("Time to execute testKDD : %f\n", elapsedTime);

    }

    MPI_Finalize();

}