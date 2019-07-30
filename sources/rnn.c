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
int nbProcs, rank;

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
        if (rank == 0){
            #pragma omp for schedule(static) private(i)
            for (i=0; i<nbColMatrix; i++) {
                layer->value[i] = matrix[nbColMatrix*raw+i];
            }
        }
        // Init weight 
        layer->weight = malloc(sizeof(double)*nbColMatrix*layerSize[currentLayer+1]);
        if (rank == 0){
            #pragma omp for schedule(static) private(i)
            for (i = 0; i < nbColMatrix*layerSize[currentLayer+1]; ++i){
                layer->weight[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
            }
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
        
        if(rank == 0){
            #pragma omp for schedule(static) private(i)
            for (i = 0; i < layerSize[currentLayer+1]*layerSize[currentLayer]; ++i){
                layer->weight[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
            }
        }
    }

    layer->bias = malloc(sizeof(double)*layerSize[currentLayer]);
    layer->value_prev = malloc(sizeof(double)*layerSize[currentLayer]);
    layer->error = malloc(sizeof(double)*layerSize[currentLayer]);
    layer->error_prev = malloc(sizeof(double)*layerSize[currentLayer]);
    
    if (rank == 0){
        #pragma omp for schedule(static) private(i)
        for (i = 0; i < layerSize[currentLayer]; ++i){
            layer->bias[i] = 1.0 * (double)rand() / RAND_MAX - 1.0/2;
            layer->value_prev[i] = 0.0;
            layer->error[i] = 0.0;
            layer->error_prev[i] = 0.0;
        }
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
    // double tmp;

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

// Unused function
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

// void sendTabLayer(int sizeOfTableOutput){

//     // typedef struct layer LAYER;
//     // struct layer {
//     //   int typeLayer;
//     //   int nbNodes;
//     //   double* bias;
//     //   double* value;
//     //   // double* value_prev;
//     //   double* error;
//     //   // double* error_prev;
//     //   double* weight;
//     // };

//     int i;
//     MPI_Datatype structureLayerMPI;
//     int structlen = 6;
//     // int blocklengths[structlen]; 
//     MPI_Datatype types[structlen];
//     MPI_Aint displacements[structlen];

//     for(i = 0; i < NBLAYER; i++){
//         // where are the components relative to the structure
//         int layerSizeI = layerSize[i];
//         int layerSizeII = layerSize[i+1]*layerSize[i];
//         int blockcounts[6] = {1,1,layerSizeI,layerSizeI,layerSizeI,layerSizeII};
//         MPI_Address(&tabLayer[i]->typeLayer, &displacements[0]);
//         MPI_Address(&tabLayer[i]->nbNodes, &displacements[1]);
//         MPI_Address(&tabLayer[i]->bias, &displacements[2]);
//         MPI_Address(&tabLayer[i]->value, &displacements[3]);
//         MPI_Address(&tabLayer[i]->error, &displacements[4]);
//         MPI_Address(&tabLayer[i]->weight, &displacements[5]);

//         // blocklengths[0] = 1; 
//         types[0] = MPI_INT;
//         // displacements[0] = (size_t)&(tabLayer[i]->typeLayer) - (size_t)&tabLayer[i];

//         // blocklengths[1] = 1; 
//         types[1] = MPI_INT;
//         // displacements[1] = (size_t)&(tabLayer[i]->nbNodes) - (size_t)&tabLayer[i];
    
//         // blocklengths[2] = layerSize[i]; 
//         types[2] = MPI_DOUBLE;
//         // displacements[2] = (size_t)&(tabLayer[i]->bias[0]) - (size_t)&tabLayer[i];

//         // blocklengths[3] = layerSize[i]; 
//         types[3] = MPI_DOUBLE;
//         // displacements[3] = (size_t)&(tabLayer[i]->value[0]) - (size_t)&tabLayer[i];

//         // blocklengths[4] = layerSize[i]; 
//         // types[4] = MPI_DOUBLE;
//         // displacements[4] = (size_t)&(tabLayer[i]->value_prev[0]) - (size_t)&tabLayer[i];

//         // blocklengths[4] = layerSize[i]; 
//         types[4] = MPI_DOUBLE;
//         // displacements[4] = (size_t)&(tabLayer[i]->error[0]) - (size_t)&tabLayer[i];

//         // blocklengths[6] = layerSize[i]; 
//         // types[6] = MPI_DOUBLE;
//         // displacements[6] = (size_t)&(tabLayer[i]->error_prev[0]) - (size_t)&tabLayer[i];

//         // blocklengths[5] = layerSize[i+1]*layerSize[i];
//         types[5] = MPI_DOUBLE;
//         // displacements[5] = (size_t)&(tabLayer[i]->weight[0]) - (size_t)&tabLayer[i];
        
//         for (i = 5; i >= 0; i--)
//             displacements[i] -= displacements[0];

//         printf("One layer ready to create struct\n");
//         MPI_Type_struct(6, blockcounts, displacements, types, &structureLayerMPI);
//         // MPI_Type_create_struct(structlen,blocklengths,displacements,types,&structureLayerMPI);
//         printf("struct created\n");
//         MPI_Type_commit(&structureLayerMPI);
//         printf("commited rank = %d \n", rank);

//         // MPI_Aint typesize;
//         // MPI_Type_extent(structureLayerMPI,&typesize);
//         // printf("extented\n");
        
//         // if (rank == 0) {
//         //     // MPI_Bcast(tabLayer, NBLAYER, MPI_INT, 0, MPI_COMM_WORLD);
//         //     MPI_Send(&tabLayer[i],1,structureLayerMPI,1,0,MPI_COMM_WORLD);
//         // }else{
//         //     MPI_Recv(&tabLayer[i],1,structureLayerMPI,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
//         // }

//         MPI_Bcast(&tabLayer[i], 1, structureLayerMPI, 0, MPI_COMM_WORLD);
//         printf("Send !\n");
//         MPI_Type_free(&structureLayerMPI);
//     }
// }

// void sendTabLayer2(int sizeOfTableOutput){
//     int i;
//     for(i = 1; i < NBLAYER; i++){
//         int layerSizeI = layerSize[i];
//         int layerSizeII = layerSize[i+1]*layerSize[i];
//         MPI_Bcast(&tabLayer[i]->typeLayer, 1, MPI_INT, 0, MPI_COMM_WORLD);
//         MPI_Bcast(&tabLayer[i]->nbNodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
//         MPI_Bcast(&tabLayer[i]->bias, layerSizeI, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//         MPI_Bcast(&tabLayer[i]->value, layerSizeI, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//         MPI_Bcast(&tabLayer[i]->error, layerSizeI, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//         MPI_Bcast(&tabLayer[i]->weight, layerSizeII, MPI_DOUBLE, 0, MPI_COMM_WORLD);

//         // if(rank == 0){
//             // send
//             // MPI_Send(&tabLayer[i]->value,1,structureLayerMPI,1,0,MPI_COMM_WORLD);
//         // }else{
//             // receive
//         // }
//     }
// }

// struct.c
// struct object {
//     char c;
//     double x[2];
//     int i;
// };
// MPI_Datatype newstructuretype;
// int structlen = 3;
// int blocklengths[structlen]; MPI_Datatype types[structlen];
// MPI_Aint displacements[structlen];
// // where are the components relative to the structure?
// blocklengths[0] = 1; types[0] = MPI_CHAR;
// displacements[0] = (size_t)&(myobject.c) - (size_t)&myobject;
// blocklengths[1] = 2; types[1] = MPI_DOUBLE;
// displacements[1] = (size_t)&(myobject.x[0]) - (size_t)&myobject;
// blocklengths[2] = 1; types[2] = MPI_INT;
// displacements[2] = (size_t)&(myobject.i) - (size_t)&myobject;
// MPI_Type_create_struct(structlen,blocklengths,displacements,types,&newstructuretype);
// MPI_Type_commit(&newstructuretype);
// {
// MPI_Aint typesize;
// MPI_Type_extent(newstructuretype,&typesize);
// if (procno==0) printf("Type extent: ");
// }
// if (procno==sender) {
// MPI_Send(&myobject,1,newstructuretype,the_other,0,comm);
// } else if (procno==receiver) {
// MPI_Recv(&myobject,1,newstructuretype,the_other,0,comm,MPI_STATUS_IGNORE);
// }
// MPI_Type_free(&newstructuretype);


void sendTabLayer(){

    // MPI_Barrier(MPI_COMM_WORLD);

    // Strategy with broadcast
    // for(i = 0; i < NBLAYER; i++){
    //     printf("Layer size : rank %d = %d\n",rank,layerSize[0] );
    //     MPI_Bcast(tabLayer[i]->value, layerSize[i], MPI_DOUBLE, 0, MPI_COMM_WORLD);
    //     if (rank != 0){
    //         printf("RANK 2\n");
    //     }
    //     printf("value : %f rank : %d\n", tabLayer[i]->value[0], rank);
    // }

    // Strategy with send and receive
    // for(i = 0; i < 1; i++){
    //     printf("Layer size : rank %d = %d\n",rank,layerSize[0] );
    //     if (rank == 0) {
    //         int j;
    //         for (j = 0; j < nbProcs; j++) {
    //             if (j != rank) {
    //                 MPI_Send(tabLayer[i]->value, layerSize[i], MPI_DOUBLE, j, 0, MPI_COMM_WORLD);
    //             }
    //         }
    //     } else {
    //         MPI_Recv(tabLayer[i]->value, layerSize[i], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
    //         printf("Receive layer : %d\n", i);
    //     }
    //     MPI_Barrier(MPI_COMM_WORLD);
    //     if (rank != 0){
    //         // tabLayer[i]->value[layerSize[i]] = 120.0;
    //         printf("RANK 2\n");
    //     }
    //     printf("value : %f rank : %d\n", tabLayer[i]->value[layerSize[i]], rank);
    // }

    // printf("Begin send tablayer : rank = %d \n", rank);
    // sendTabLayer(sizeOfTableOutput);
    // sendTabLayer2(sizeOfTableOutput);
    // printf("NBLAYER = %d  : rank = %d \n",NBLAYER, rank);
    // double* tabTmp;
    // for(i = 0; i < NBLAYER; i++){
    //     printf("Layer : %d\n", i);
    //     int layerSizeI = layerSize[i];

        // if (i != NBLAYER-1){
        //     printf("Hidden Layer\n");
        //     int layerSizeII = layerSize[i+1]*layerSize[i];
        //     printf("layerSizeII : %d rank : %d\n", layerSizeII, rank);
        //     // MPI_Barrier(MPI_COMM_WORLD);

        //     // Strategy with broadcast routine
        //     // if (rank != 0){
        //         // free(tabLayer[i]->weight);
        //         // tabLayer[i]->weight = malloc(sizeof(double)*layerSizeII);
        //     // }

        //     // MPI_Barrier(MPI_COMM_WORLD);
            // MPI_Bcast(&tabLayer[i]->weight, layerSizeII, MPI_DOUBLE, 0, MPI_COMM_WORLD);            
           
        //     // Strategy with Send/Receive routines
        //     // if (rank == 0) {
        //     //     int j;
        //     //     for (j = 0; j < nbProcs; j++) {
        //     //         if (j != rank) {
        //     //             MPI_Send(&tabLayer[i]->weight, layerSize[i+1]*layerSize[i], MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
        //     //             printf("weight : %f rank : %d\n", tabLayer[i]->weight[0], rank);
        //     //         }
        //     //     }
        //     // }else{
        //     //     // free(tabLayer[i]->weight);
        //     //     tabLayer[i]->weight = malloc(sizeof(double)*layerSize[i+1]*layerSize[i]);
        //     //     MPI_Recv(&tabLayer[i]->weight, layerSize[i+1]*layerSize[i], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //     //     printf("weight : %f rank : %d\n", tabLayer[i]->weight[0], rank);
        //     // }
        // }       

        // printf("!!!!! RANK %d !!!!! bias : %f\n", rank, tmp);
        // MPI_Bcast(tabLayer[i]->bias, layerSizeI, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // SEND RECEIVE STRATEGY BIAS
        // if (rank == 0) {
        //     int j;
        //     for (j = 1; j < nbProcs; j++) {
        //         printf("layer size : %d\n", layerSize[i]);
        //         int ierr = MPI_Send(tabLayer[i]->bias, layerSize[i], MPI_DOUBLE, j, 0, MPI_COMM_WORLD);
        //         printf("ierr : %d \n", ierr);
        //         if (ierr != MPI_SUCCESS) {
        //            printf("MERDE\n");
        //        }
        //         printf("bias : %f rank : %d\n", tabLayer[i]->bias[0], rank);
        //     }
        // }else{
        //     // free(tabLayer[i]->weight);
        //     // tabLayer[i]->weight = malloc(sizeof(double)*layerSize[i]);
        //     printf("Receive\n");
        //     int c = MPI_Recv(tabLayer[i]->bias, layerSize[i], MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        //     printf("c : %d \n", c);
        //     printf("bias : %f rank : %d\n", tabLayer[i]->bias[0], rank);
        // }

        // printf("bias : %f rank : %d\n", tabLayer[i]->bias[0], rank);
        
        // MPI_Bcast(&tabLayer[i]->typeLayer, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // printf("typeLayer : %d rank : %d\n", tabLayer[i]->typeLayer, rank);

        // MPI_Bcast(&tabLayer[i]->nbNodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // printf("nbNodes : %d rank : %d\n", tabLayer[i]->nbNodes, rank);

        // MPI_Bcast(&tabLayer[i]->value, layerSizeI, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // printf("value : %f rank : %d\n", tabLayer[i]->value[0], rank);

        // MPI_Bcast(&tabLayer[i]->error, layerSizeI, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // printf("error : %f rank : %d\n", tabLayer[i]->error[0], rank);

        // tabTmp = malloc(sizeof(double)*layerSize[i]);
        // if (rank == 0) {
        //     for(int j = 0; j < layerSize[i]; j++){
        //         tabTmp[j] = tabLayer[i]->bias[j];
        //     }
        // }

        // MPI_Bcast(&tabTmp, layerSize[i], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // if (rank != 0){
        //     printf("FIll Tab : %d \n",layerSize[i]);
        //     for(int j = 0; j < layerSize[i]; j++){
        //         tabLayer[i]->bias[j] = tabTmp[j];
        //     }
        //     printf("END FILL TAB!!!!!!!!!!!\n");
        // }

        // MPI_Bcast(layerSize, NBLAYER, MPI_INT, 0, MPI_COMM_WORLD);
        // MPI_Bcast(tabLayer[i]->bias, layerSize[i], MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // if (rank != 0){
        //     printf("RANK 2\n");
        // }
        // printf("last bias : %f | rank : %d\n", tabLayer[i]->bias[layerSize[i]], rank);
        // double tmp;
        // for(int j = 0; j < layerSize[i]; j++){
        //     if (rank == 0){
        //         tmp = tabLayer[i]->bias[j];
        //     }
        //     MPI_Bcast(&tmp, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        //     printf("Bias send : bias[j] = %f | rank = %d\n", tabLayer[i]->bias[j], rank);
        //     tabLayer[i]->bias[j] = tmp;
        // }
        // printf("Bias send : bias[1] = %f | rank = %d\n", tabLayer[i]->bias[0], rank);

        // printf("end Layer : %d\n", i);

        // if(rank == 0){
            // send
            // MPI_Send(&tabLayer[i]->value,1,structureLayerMPI,1,0,MPI_COMM_WORLD);
        // }else{
            // receive
        // }
    // }
    
    // free(tabTmp);

    // if (rank == 1){
    //     printf("bias : %f rank : %d\n", tabLayer[i]->bias[0], rank);
    // }

    // printf("End of send everything : rank = %d \n", rank);

    double * tmp;
    int j,i;
    for(j = 0; j < NBLAYER-1; j++){
        // outTable
        // nbErrorFind
        // nbOutTable
        // nbErrorFind
        // nbOutTable

        // Send Matrix
        if (rank != 0){
            matrix = malloc(sizeof(double)*nbColMatrix*nbRawMatrix);
            vectorOutput = malloc(sizeof(int)*nbRawMatrix);
        }
        MPI_Bcast(matrix, nbColMatrix*nbRawMatrix, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(vectorOutput, nbRawMatrix, MPI_INT, 0, MPI_COMM_WORLD);

        // Send typeLayer
        MPI_Bcast(&tabLayer[j]->typeLayer, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // printf("typeLayer : %d rank : %d\n", tabLayer[j]->typeLayer, rank);
        
        // Send nbNodes        
        MPI_Bcast(&tabLayer[j]->nbNodes, 1, MPI_INT, 0, MPI_COMM_WORLD);
        // printf("nbNodes : %d rank : %d\n", tabLayer[j]->nbNodes, rank);

        // Send value
        tmp = malloc(sizeof(double)* layerSize[j]);
        if (rank == 0){
            for(i = 0; i < layerSize[j]; i++){
                tmp[i] = tabLayer[j]->value[i];
            }
        }
        MPI_Bcast(tmp, layerSize[j], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank != 0){
            for(i = 0; i < layerSize[j]; i++){
                tabLayer[j]->value[i] = tmp[i];
            }
        }
        // printf("value[0] : %f rank : %d\n", tabLayer[j]->value[0], rank);
        
        // Send error
        if (rank == 0){
            for(i = 0; i < layerSize[j]; i++){
                tmp[i] = tabLayer[j]->error[i];
            }
        }
        MPI_Bcast(tmp, layerSize[j], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank != 0){
            for(i = 0; i < layerSize[j]; i++){
                tabLayer[j]->error[i] = tmp[i];
            }
        }
        // printf("layer : %d error[0] : %f rank : %d\n",j, tabLayer[j]->error[0], rank);

        // Send bias
        if (rank == 0){
            for(i = 0; i < layerSize[j]; i++){
                tmp[i] = tabLayer[j]->bias[i];
            }
        }
        MPI_Bcast(tmp, layerSize[j], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank != 0){
            for(i = 0; i < layerSize[j]; i++){
                tabLayer[j]->bias[i] = tmp[i];
            }
        }
        // printf("layer : %d bias[0] : %f rank : %d\n",j, tabLayer[j]->bias[0], rank);

        // Send weight
        tmp = malloc(sizeof(double)* layerSize[j]*layerSize[j+1]);
        if (rank == 0){
            for(i = 0; i < layerSize[j]*layerSize[j+1]; i++){
                tmp[i] = tabLayer[j]->weight[i];
            }
        }
        MPI_Bcast(tmp, layerSize[j]*layerSize[j+1], MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank != 0){
            for(i = 0; i < layerSize[j]*layerSize[j+1]; i++){
                tabLayer[j]->weight[i] = tmp[i];
            }
        }
        // printf("layer : %d bias[0] : %f rank : %d\n",j, tabLayer[j]->weight[0], rank);
    }
}

void shareErrorFind(int * nbErrorFind){
    if (nbProcs >1){
        if (rank != 0) {
            MPI_Send(nbErrorFind, sizeOfTableOutput, MPI_INT, 0, 123, MPI_COMM_WORLD);
        } else {
            int i;
            int * tmp = malloc(sizeof(int)*sizeOfTableOutput);
            MPI_Recv(tmp, sizeOfTableOutput, MPI_INT, MPI_ANY_SOURCE, 123, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            for(i = 0; i < sizeOfTableOutput; i++){
                nbErrorFind[i] += tmp[i];
            }
        }
    }
}

void testKDD(){
    // Block the caller until all processes in the communicator have called it
    MPI_Barrier(MPI_COMM_WORLD);

    int i;
    if (rank == 0){
        printf("Init Test\n");
        nbRawMatrix = 0;
        preprocessing(fileTestKdd, 1);
        initLayerSize();
    }

    MPI_Bcast(&nbColMatrix, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(layerSize, NBLAYER, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&sizeOfTableOutput, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nbRawMatrix, 1, MPI_INT, 0, MPI_COMM_WORLD);
   
    if (rank != 0){
        for (i = 0; i < NBLAYER; ++i){
            tabLayer[i] = (LAYER*)malloc(sizeof(LAYER));
        }
        for (i = 0; i < NBLAYER; ++i){
            init_layer(tabLayer[i], 0, i);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    sendTabLayer();

    if(rank == 0){
        printf("! Structure Sent Successfully !\n");
        printf("Begin test\n");
    }

    nbErrorFind = malloc(sizeof(int)*sizeOfTableOutput);
    for (i = 0; i < sizeOfTableOutput; ++i){
        nbErrorFind[i] = 0;
    }    
    
    // Method to split loop

    // 1 :
    // NLocal = N/nbProcs
    // for(i=rank*NLocal;i<(rank+1)*NLocal;i++)
    
    // 2 :
    // if (i%nbProcs != rank) continue;

    // RUN
    double error = 1.0;
    ((void) error);
    double * outc = malloc(sizeof(double)*sizeOfTableOutput);
    for (i = 0; i < nbRawMatrix; ++i){
        if (i%nbProcs != rank) continue;
        error = 1.0;
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
    MPI_Barrier(MPI_COMM_WORLD);
    // Send nbErrorFind, add every error to display Error Find
    // printf("Error find by rank : %d \n", rank);
    // displayErrorFindRank(nbErrorFind, rank);
    
    if(rank == 0){
        int i;
        for(i = 0; i < nbProcs-1; i++){
            shareErrorFind(nbErrorFind);    
        }
    }else{
        shareErrorFind(nbErrorFind);    
    }

    if(rank == 0){
        printf("\nError Find in TestKDD\n");
        displayErrorFind(nbErrorFind);
    }
}

int main(int argc, char ** argv){
    // clock_t start, end;
    double tbeg=0, cpu_time_used=0;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &nbProcs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0){
        printf("Begin Learn\n");
        tbeg = MPI_Wtime();
        // start = clock();
        learnKDD();
        // end = clock();
        // cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
        cpu_time_used = MPI_Wtime() - tbeg;
        printf("Time to execute learnKDD : %f\n", cpu_time_used);

        printf("End Learn, begin test\n");
        // start = clock();
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        tbeg = MPI_Wtime();
        

    testKDD();

    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        cpu_time_used = MPI_Wtime() - tbeg;
        printf("Time to execute testKDD : %f | rank : %d\n", cpu_time_used, rank);
    }
    MPI_Finalize();
    return 1;
}