
#define NBCOL 41
#define NBLAYER 5
#define NBHD 120

void displayErrorFind();
char* getfield(char*, int);
void readFile(char *, char**);
void makeVector(int, char *, double*, int*, char**);
void makeMatrix(char *, double*, int*, char**);
int getNbColMatrix();
int getNbRawMatrix();
int getSizeOfTableOutput();