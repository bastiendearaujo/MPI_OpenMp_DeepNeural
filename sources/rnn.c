#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <math.h>
#include <assert.h>
#include <time.h>
#include <omp.h>

#include "utils.h"
#include "preprocessing.h"

#define NUM_THREADS 2

char * fileLearnKdd = "./exampleFiles/KDDTest+.txt";
char * fileTestKdd = "./exampleFiles/KDDTest+.txt";

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

int * nbErrorFind;
double * matrix;
char * outTableRnn [5] = {"Normal", "Probe", "DOS", "R2L", "U2R"};
int * vectorOutput;
int nbColMatrix;
int nbRawMatrix;
int sizeOfTableOutput;

// MPI VARIABLE
int initialized, finalized;
int nb_procs, rank;

void preprocessing(char * nameFile, int isTest){
    readFile(nameFile);

    sizeOfTableOutput = getSizeOfTableOutput();

    nbColMatrix = getNbColMatrix();
    nbRawMatrix = getNbRawMatrix();

    matrix = malloc(sizeof(double)*nbColMatrix*nbRawMatrix);
    vectorOutput = malloc(sizeof(int)*nbRawMatrix);


    makeMatrix(nameFile, matrix, vectorOutput);
}

void init_layer(LAYER * layer, int raw, int currentLayer){
    layer->typeLayer = currentLayer;

    int i;
    // Init Input Layer
    if (currentLayer == 0){
        layer->nbNodes = nbColMatrix;
        // Init vector input layer
        layer->value = malloc(sizeof(double)*nbColMatrix);
        #pragma omp for schedule(static) private(i)
        for (i=0; i<nbColMatrix; i++) {
            layer->value[i] = matrix[nbColMatrix*raw+i];
        }
        // Init weight 
        layer->weight = malloc(sizeof(double)*nbColMatrix*layerSize[currentLayer+1]);
        #pragma omp for schedule(static) private(i)
        for (i = 0; i < nbColMatrix*layerSize[currentLayer+1]; ++i){
            layer->weight[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
        }
    }else if (currentLayer == NBLAYER-1){ // Init Output Layer
        layer->nbNodes = layerSize[currentLayer];
        layer->value = malloc(sizeof(double)*layerSize[currentLayer]);
    }else{  // Init Hidden Layer
        layer->nbNodes = layerSize[currentLayer];
        layer->value = malloc(sizeof(double)*layerSize[currentLayer]);
        #pragma omp for schedule(static) private(i)
        for (i=0; i<layerSize[currentLayer]; i++) {
            layer->value[i] = 0.0;
        }

        layer->weight = malloc(sizeof(double) * layerSize[currentLayer+1] * layerSize[currentLayer]);
        

        #pragma omp for schedule(static) private(i)
        for (i = 0; i < layerSize[currentLayer+1]*layerSize[currentLayer]; ++i){
            layer->weight[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
        }
    }

    layer->bias = malloc(sizeof(double)*layerSize[currentLayer]);
    #pragma omp for schedule(static) private(i)
    for (i = 0; i < layerSize[currentLayer]; ++i){
        layer->bias[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
    }

    layer->value_prev = malloc(sizeof(double)*layerSize[currentLayer]);
    #pragma omp for schedule(static) private(i)
    for (i=0; i<layerSize[currentLayer]; i++) {
        layer->value_prev[i] = 0.0;
    }

    layer->error = malloc(sizeof(double)*layerSize[currentLayer]);
    #pragma omp for schedule(static) private(i)
    for (i=0; i<layerSize[currentLayer]; i++) {
        layer->error[i] = 0.0;
    }

    layer->error_prev = malloc(sizeof(double)*layerSize[currentLayer]);
    #pragma omp for schedule(static) private(i)
    for (i=0; i<layerSize[currentLayer]; i++) {
        layer->error_prev[i] = 0.0;
    }
}

void intiValueLayer(LAYER * layer, int raw){
    int i;
    #pragma omp for schedule(static) private(i)
    for (i=0; i<layerSize[0]; i++) {
        layer->value[i] = matrix[nbColMatrix*raw+i];
    }
}

// init value prev
void rnnsetstart(LAYER * tabLayer[]){
    int i, k;
    #pragma omp for schedule(static) private(i)
    for (i = 0; i < NBLAYER; ++i){
        if (tabLayer[i]->typeLayer == NBLAYER-1){ // Output
            for (k = 0; k < tabLayer[i]->nbNodes; ++k){
                tabLayer[i]->value_prev[k] = tanh(tabLayer[i]->value[k]);
            }
        }else{ // Not Output
            for (k = 0; k < tabLayer[i]->nbNodes; ++k){
                tabLayer[i]->value_prev[k] = tabLayer[i]->value[k];
            }
        }
    }
}

// init value
void rnnset(LAYER * tabLayer[], double * out){
    
    int i,j;
    #pragma omp for schedule(static) private(i)
    for (i = 0; i < NBLAYER; ++i){
        // If not input
        if(tabLayer[i]->typeLayer != 0){
            for (j = 0; j < tabLayer[i]->nbNodes; ++j){
                tabLayer[i]->value[j] = tabLayer[i]->bias[j];
            }

            matrixTimesMatrixTan(tabLayer[i-1]->value, tabLayer[i-1]->weight, tabLayer[i]->value, tabLayer[i-1]->nbNodes, tabLayer[i]->nbNodes);
        }

        for (j = 0; j < tabLayer[i]->nbNodes; ++j){
            tabLayer[i]->error[j] = 0.0;
            tabLayer[i]->value_prev[j] = 0.0;
        }
    }

}

void rnnlearn(LAYER * tabLayer[], double * out, double learningrate){
    int i,j,k;
    double normalize = 5.0;
    double * deltaWeight;
    double tmp;

    for(i = NBLAYER-2; i>= 0; i--){
        if (i == NBLAYER-2){
            vectorSubstraction(tabLayer[NBLAYER-1]->value, out, tabLayer[NBLAYER-1]->error, tabLayer[NBLAYER-1]->nbNodes);
        }
        for(j = 0; j < tabLayer[i]->nbNodes; j++){
            deltaWeight = malloc(sizeof(double)*tabLayer[i]->nbNodes * tabLayer[i+1]->nbNodes);
            for (k = 0; k < tabLayer[i+1]->nbNodes; ++k){
                deltaWeight[j*tabLayer[i+1]->nbNodes+k] = -(tabLayer[i+1]->error[k]*(tabLayer[i]->value[j])*learningrate);
                tabLayer[i]->error[j] += tabLayer[i]->weight[j*tabLayer[i+1]->nbNodes+k]*tabLayer[i+1]->error[k];
                tabLayer[i]->weight[j*tabLayer[i+1]->nbNodes+k] += deltaWeight[j*tabLayer[i+1]->nbNodes+k];
                
                // Update BIAS
                // if(i != NBLAYER-2){
                //     tmp = (1 - tabLayer[i+1]->value[k])*tabLayer[i+1]->value[k];
                //     tmp *= learningrate * tabLayer[i+1]->error[k];
                //     tabLayer[i+1]->bias[k] -= tmp;

                //     if(tabLayer[i]->bias[k]>normalize){
                //         tabLayer[i]->bias[k]=normalize;
                //     }else if(tabLayer[i]->bias[k]< (-normalize)){
                //         tabLayer[i]->bias[k]=(-normalize);
                //     } 

                // }else{
                //     tabLayer[i+1]->bias[k] -= learningrate * tabLayer[i+1]->error[k];
                // }

                if(tabLayer[i]->weight[j*tabLayer[i+1]->nbNodes+k]>normalize){
                    tabLayer[i]->weight[j*tabLayer[i+1]->nbNodes+k]=normalize;
                }else if(tabLayer[i]->weight[j*tabLayer[i+1]->nbNodes+k]<(-normalize)){
                    tabLayer[i]->weight[j*tabLayer[i+1]->nbNodes+k] = (-normalize);
                }
            }

            free(deltaWeight);
            tabLayer[i]->error[j] *= tabLayer[i]->value[j]*(1 - tabLayer[i]->value[j]);  
        }
    }
}

double geterror(LAYER* layer, double * out){
    double res =0.0, tmp;
    int i;
    #pragma omp for schedule(static) private(i)
    for (i = 0; i < layer->nbNodes; ++i){ 
        tmp = out[i]-layer->value[i];
        if (tmp<0){
            res -= tmp;
        }else{
            res += tmp;
        }
    }
    return res/(layer->nbNodes);
}

void displayResult(LAYER * layer){
    int i;
    for (i = 0; i < layer->nbNodes; ++i){ 
        printf("Result [%s] = %f\n",outTableRnn[i], layer->value[i]);
    }
}

void fillOutc(double * out, int raw){
    int i; 
    // #pragma omp for schedule(static) private(i)
    for (i = 0; i < sizeOfTableOutput; ++i){
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
    #pragma omp for schedule(static) private(i)
    for (i = 1; i < NBLAYER-1; ++i){
        layerSize[i] = NBHD;
    }
    layerSize[NBLAYER-1] = sizeOfTableOutput;
}

void wichError(LAYER * layer, double * out){
    double max = 0.0;
    char * res;
    ((void) res);
    int i, indiceMax =0, indiceOut = 0;
    #pragma omp for schedule(static) private(i)
    for (i = 0; i < layer->nbNodes; ++i){ 
        if (layer->value[i]>max){
            max = layer->value[i];
            res = outTableRnn[i];
            indiceMax = i;
        }

        if (out[i] == 1){
            indiceOut = i;
        }
    }

    if ((indiceOut == 0 && indiceMax == 0) || (indiceOut > 0 && indiceMax > 0)){
        nbErrorFind[indiceMax]++; 
    }

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
    #pragma omp for schedule(static) private(i)
    for (i = 0; i < layer->nbNodes; i++){ 
        sum += layer->value[i];
    }

    for (i = 0; i < layer->nbNodes; i++){ 
        layer->value[i] /= sum;
    }
}

// Create new layer to send at each cluster
void splitLayer(int rank, LAYER * oldTabLayer[], LAYER * newTabLayer[]){

    /***** Split INPUT layer *****/
    int nbElemInput, i, j, k;
    
    // Split value
    if (rank != 11){
        nbElemInput = 10;

    }else{
        nbElemInput = 12;
    } 
    newTabLayer[0]->nbNodes = nbElemInput;
    newTabLayer[0]->typeLayer = oldTabLayer[0]->typeLayer;
    for (i = 0; i < nbElemInput; i++){
        newTabLayer[0]->value[i] = oldTabLayer[0]->value[10*rank+i];
    }

    /**** Split HD layer ****/
    for(i = 1; i < NBLAYER-2; i++){ //Each hidden layer
        newTabLayer[i]->nbNodes = 10;
        // Split value
        for (j = 0; j < 10; j++){
            newTabLayer[i]->value[j] = oldTabLayer[i]->value[10*rank+j];
        }

        // Split weight
        for (k = 0; k < tabLayer[k]->nbNodes; k++) {
            for (j = 0; j < tabLayer[i-1]->nbNodes; j++) {
                newTabLayer[i]->weight[k] = oldTabLayer[i]->weight[j*tabLayer[i]->nbNodes+k];
            }
        }
    }
}

void learnKDD(){
    int i, j, sum=0;
    clock_t start, end;
    double cpu_time_used;
    nbRawMatrix = 0;
    printf("Begin preprocessing\n");
    start = clock();
    preprocessing(fileLearnKdd, 0);
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time to execute preprocessing : %f\n", cpu_time_used);

    nbErrorFind = malloc(sizeof(int)*sizeOfTableOutput);
    #pragma omp for schedule(static) private(i)
    for (i = 0; i < sizeOfTableOutput; ++i){
        nbErrorFind[i] = 0;
    }

    initLayerSize();

    for (i = 0; i < NBLAYER; ++i){
        tabLayer[i] = (LAYER*)malloc(sizeof(LAYER));
    }
    for (i = 0; i < NBLAYER; ++i){
        init_layer(tabLayer[i], 0, i);
    }
    
    // RUN
    double error = 1.0;
    ((void) error);
    double * out = malloc(sizeof(double)*sizeOfTableOutput);
    double learningrate = 0.1;

    for (i = 0; i < nbRawMatrix; ++i){
        error = 1.0;
        j = 0;
        intiValueLayer(tabLayer[0], i);
        fillOutc(out, i);
        while (error > 0.05 && j<100) {
            rnnsetstart(tabLayer);
            rnnset(tabLayer, out);
            ajustError(tabLayer[NBLAYER-1]);
            rnnlearn(tabLayer,out,learningrate);
            error = geterror(tabLayer[NBLAYER-1], out);
            j++;
            

        }
        #ifdef ALLINF      
            printf("\nLigne : %d | trouvé en : %d iteration  | ", i+1, j);
            displayVector(out, sizeOfTableOutput);
        #endif

        wichError(tabLayer[NBLAYER-1], out);
        #ifdef ALLINF      
            displayResult(tabLayer[NBLAYER-1]);
        #endif

        sum+=j;

    }

    
    free(out);

    printf("\nMoyenne iteration par ligne  : %d \n", sum / nbRawMatrix);

    printf("\nError Find in learnKDD : \n");
    displayErrorFind(nbErrorFind);
}

void testKDD(){
    nbRawMatrix = 0;
    preprocessing(fileTestKdd, 1);

    nbErrorFind = malloc(sizeof(int)*sizeOfTableOutput);
    for (int i = 0; i < sizeOfTableOutput; ++i){
        nbErrorFind[i] = 0;
    }

    initLayerSize();
    // RUN
    double error = 1.0;
    ((void) error);
    double * outc = malloc(sizeof(double)*sizeOfTableOutput);
    int i;
    for (i = 0; i < nbRawMatrix; ++i){
        error = 1.0;
        // init_layer(tabLayer[0], i, 0);
        intiValueLayer(tabLayer[0], i);
        fillOutc(outc, i);
        
        rnnsetstart(tabLayer);
        rnnset(tabLayer, outc);
        ajustError(tabLayer[NBLAYER-1]);
        error = geterror(tabLayer[NBLAYER-1], outc);

        #ifdef ALLINF      
            printf("\nLigne : %d | Error : %f |", i, error);
            displayVector(outc, sizeOfTableOutput);
        #endif

        wichError(tabLayer[NBLAYER-1], outc);
        #ifdef ALLINF      
            displayResult(tabLayer[NBLAYER-1]);
        #endif


        // freeLayer(tabLayer[0]);
    }

    free(outc);
    printf("\nError Find in TestKDD : \n");
    displayErrorFind(nbErrorFind);
}

int main(int argc, char ** argv){
    // double tbeg, elapsedTime;

    // MPI_Init(&argc, &argv);
    // MPI_Comm_size(MPI_COMM_WORLD, &nb_procs);
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    printf("Begin Learn\n");
    // tbeg = MPI_Wtime();
    clock_t start, end;
    double cpu_time_used;
    start = clock();
    learnKDD();
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    // elapsedTime = MPI_Wtime() - tbeg;
    printf("Time to execute learnKDD : %f\n", cpu_time_used);

    printf("End Learn, begin test\n");
    start = clock();
    testKDD();
    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time to execute testKDD : %f\n", cpu_time_used);
}