#include <stdio.h>
#include <math.h>
#include <string.h>

// UTILS TABLE

void printTab(char** tab, int size){
    printf("\n[");
    for (int i = 0; i < size; ++i){
        printf("%s ;", tab[i]);
    }
    printf("]\n");
}

int isPresentTab(char ** tab, int size, char * str){
    int i;
    // printf("TOP\n");
    for(i = 0; i < size; i++){
        // printf("tab[i] %s = %s \n",tab[i], str);
        if(!strcmp(tab[i], str)){
            return 1;
        }
    }
    return 0;
}

// UTILS MATRIX AND VECTOR
void displayMatrix(double* matrix, int nbRaw, int nbCol){
    int i,j;
    for (i=0; i<nbRaw; i++) {
        for (j=0; j<nbCol; j++) {
            printf("%g ",matrix[i*5+j]);
        }
        printf("\n");
    }
}

void displayVector(double * vector, int nbRaw){
    printf("[");
    for (int i = 0; i < nbRaw; ++i){
        printf("%f ; ",vector[i]);
    }
    printf("]\n");
}

double sigmoid(double a){
    double res = 1/(1 + exp(-a));
    return res;
}

void matrixTimesMatrixTan(double * matrix_A, double * matrix_C, double * matrix_B, double * matrix_Result, double* bias, int n, int m){
    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            matrix_Result[i] += matrix_B[j*m+i] * matrix_A[j];
            matrix_Result[i] += matrix_B[j*m+i] * matrix_C[j];
        }
        matrix_Result[i] = sigmoid(matrix_Result[i]);
    }
}

void vectorSubstraction(double * a, double * b, double * c , int size){
    for (int i = 0; i < size; ++i){
        c[i] = a[i] - b[i];
    }
}

void vectorTimesDouble(double * vector, double * resultat, double x, int size, int compute){
    for (int i = 0; i < size; ++i){
        if(compute){
            resultat[i] *= vector[i] * x;
        }else{
            resultat[i] = vector[i] * x;            
        }
    }
}

void matrixScalaire(double * matrix, double * vector, int col, int raw){
    int i, j;
    for (i = 0; i < col; ++i){
        for (j = 0; j < raw; ++j){
            matrix[i*raw+j] *= vector[j];
        }
    }
}

void matrixTimesMatrix(double * matrix_A, double * matrix_B, double * matrix_Result, int n, int m){
    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            matrix_Result[i] +=  matrix_B[i*n+j]* matrix_A[j];
        }
    }
}

void matrixMinusMatrix(double * a, double * b, int raw, int col){
    int i, j;
    for (i = 0; i < raw; ++i){
        for (j = 0; j < col; ++j){
            a[i*col+j] -= b[i*col+j];
        }
    }
}

void vectorMinusVector(double * a, double * b, int col){
    int i;
    for (i = 0; i < col; ++i){
        a[i] -= b[i];
    }
}

void computeDerivative(double * derivative, double * value, int size){
    int i;
    for (i = 0; i < size; ++i){
        derivative[i] = 1.0 - (value[i]*value[i]);
    }
}

void normalisation(double* a, int interval, int size){
    int i;
    for (i = 0; i < size; ++i){
        if(a[i]>interval){
            a[i]=interval;
        }else if(a[i]<-interval){
            a[i]=-interval;
        }
    }
}