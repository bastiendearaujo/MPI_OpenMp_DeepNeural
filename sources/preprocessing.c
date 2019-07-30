#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

#include "preprocessing.h"
#include "utils.h"

int currentRaw;
int currentCol;
int nbRawMatrix;
int nbColMatrix;

int * nbOutTable;

int sizeProbe = 10;
int sizeDos = 18;
int sizeR2l = 24;
int sizeU2r = 17;

char * PROBE [10] = {"portsweep", "nmap", "ipsweep", "queso", "satan", "saint", "mscan", "ntinfoscan", "lsdomain", "illegal-sniffer"};
char * DOS [18] = {"udpstorm","apache2", "smurf", "neptune", "dosnuke", "land", "pod", "back", "teardrop", "tcpreset", "syslogd", "crashiis", "arppoison", "mailbomb", "selfping","processtable", "upstorm", "worm"};
char * R2L [24] = {"dict", "spy", "multihop", "warezmaster", "guess_passwd", "netcat", "sendmail", "imap", "ncftp", "xlock", "xsnoop", "sshtrojan", "framespoof", "ppmacro", "guest", "netbus", "snmpget", "snmpguess", "snmpgetattack", "ftp_write", "warezclient", "httptunnel", "phf", "named"};
char * U2R [17] = {"sechole", "rootkit", "buffer_overflow", "xterm", "eject", "ps", "nukepw", "secret", "perl", "yaga", "fdformat", "ffbconfig", "casesen", "ntfsdos", "ppmacro", "loadmodule", "sqlattack"};

int sizeOfTableOutput;
char * outTable [5] = {"Normal", "Probe", "DOS", "R2L", "U2R"};

int sizeOfTableProtocol = 3;
int sizeOfTableService = 70;
int sizeOfTableFlag = 11;
char * protocolTable [3] = {"tcp","udp","icmp"};
char * serviceTable[70] = {"ftp_data","other","private","http","remote_job","name","netbios_ns","eco_i","mtp","telnet","finger","domain_u",
                        "supdup","uucp_path","Z39_50","smtp","csnet_ns","uucp","netbios_dgm","urp_i","auth","domain","ftp","bgp","ldap",
                        "ecr_i","gopher","vmnet","systat","http_443","efs","whois","imap4","iso_tsap","echo","klogin","link","sunrpc","login",
                        "kshell","sql_net","time","hostnames","exec","ntp_u","discard","nntp","courier","ctf","ssh","daytime",
                        "shell","netstat","pop_3","nnsp","IRC","pop_2","printer","tim_i","pm_dump","red_i","netbios_ssn","rje","X11","urh_i",
                        "http_8001","aol","http_2784","tftp_u","harvest"};
char * flagTable [11] = {"SF","S0","REJ","RSTR","SH","RSTO","S1","RSTOS0","S3","S2","OTH"};

void displayErrorFind(int * nbErrorFind){
    int sumFind = 0;
    int sumReal = 0;
    int i;

    for (i = 0; i < sizeOfTableOutput; ++i){
        printf("%s : %d / %d | difference : %d \n", outTable[i] ,nbErrorFind[i], nbOutTable[i], nbErrorFind[i]-nbOutTable[i]);
        sumFind += nbErrorFind[i];
        sumReal += nbOutTable[i];
    }

    printf("total find : %d / %d  |  %d percent \n", sumFind, sumReal, ((100*sumFind)/sumReal));
}

void displayErrorFindRank(int * nbErrorFind, int rank){
    int sumFind = 0;
    int sumReal = 0;
    int i;
    if (rank != 0)
        nbOutTable = malloc(sizeof(int)*sizeOfTableOutput);
    MPI_Bcast(nbOutTable, sizeOfTableOutput, MPI_INT, 0, MPI_COMM_WORLD);

    for (i = 0; i < sizeOfTableOutput; ++i){
        printf("%s : %d / %d | difference : %d \n", outTable[i] ,nbErrorFind[i], nbOutTable[i], nbErrorFind[i]-nbOutTable[i]);
        sumFind += nbErrorFind[i];
        sumReal += nbOutTable[i];
    }

    printf("total find : %d / %d  |  %d percent \n", sumFind, sumReal, ((100*sumFind)/sumReal));
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

void readFile(char * nameFile){
    int i;
    nbRawMatrix = 0;
    FILE* stream = fopen(nameFile, "r");

    sizeOfTableOutput = 5;
    // int i;
    char line[1024];

    while (fgets(line, 1024, stream)){
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
    for (i = 0; i < sizeOfTableOutput; ++i){
        nbOutTable[i] = 0;
    }

    nbColMatrix = sizeOfTableProtocol + sizeOfTableService + sizeOfTableFlag + 38;
}

void makeVector(int currentColinRaw, char * field, double * matrix, int * vectorOutput){
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

        // Check if is present in one attack table (normal, probe, dos, r2l, u2r)
        if (isPresentTab(PROBE, sizeProbe, field)){
            // incremente PROBE
            nbOutTable[1]++;
            vectorOutput[currentRaw] = 1;
        } else if (isPresentTab(DOS, sizeDos, field)){
            // incremente DOS
            nbOutTable[2]++; 
            vectorOutput[currentRaw] = 2;
        }else if (isPresentTab(R2L, sizeR2l, field)){
            // incremente R2L
            nbOutTable[3]++;
            vectorOutput[currentRaw] = 3;
        }else if (isPresentTab(U2R, sizeU2r, field)){
            // incremente U2R
            nbOutTable[4]++;
            vectorOutput[currentRaw] = 4;
        }else if (!strcmp(field, "normal")){
            // incremente Normal
            nbOutTable[0]++;
            vectorOutput[currentRaw] = 0;
        }else{
            // Erreur inconnue
            printf("L'erreur : ' %s ' n'est pas prise un charge par le programme.\n Arrêt.\n", field);
            exit(0);
        }

    }else {
        matrix[currentRaw*nbColMatrix+currentCol] = tanh(atof(field));
        currentCol++;
    }

}

void makeMatrix(char * nameFile, double * matrix, int * vectorOutput){
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
            makeVector(i, field, matrix, vectorOutput);
            free(tmp);
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