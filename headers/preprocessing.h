
#define NBCOL 41
#define NBLAYER 5
#define NBHD 120

void displayErrorFind(int * nbErrorFind);
char* getfield(char*, int);
void readFile(char *);
void makeVector(int, char *, double*, int*);
void makeMatrix(char *, double*, int*);
int getNbColMatrix();
int getNbRawMatrix();
int getSizeOfTableOutput();