#ifndef __ExpMHigham__
#define __ExpMHigham__

// Coefficients kept in a namespace since no portable way to include
// as non-integer private static const data members of a class.
namespace ExpMHighamNamespace
{
    // Pade coefficients.
    static const double b1 = 64764752532480000;
    static const double b2 = 32382376266240000;
    static const double b3 = 7771770303897600;
    static const double b4 = 1187353796428800;
    static const double b5 = 129060195264000;
    static const double b6 = 10559470521600;
    static const double b7 = 670442572800;
    static const double b8 = 33522128640;
    static const double b9 = 1323241920;
    static const double b10 = 40840800;
    static const double b11 = 960960;
    static const double b12 = 16380;
    static const double b13 = 182;
    static const double b14 = 1;

    // Constant used to determine how to scale the matrix A.
    // For stability, need the norm of A to be < 5.3.
    static const double tm = 5.4;
}

class ExpMHigham
{
public:
    const int Dim;  // i.e. max size of matrices

    ExpMHigham(int maxN);  // usual constructor
    ~ExpMHigham();  // 
    double Compute(const int m, double *A);

private:
    int m;
    int mSquared;

    // Scratch arrays for computing intermediate matrices.
    double *A2;
    double *A4;
    double *A6;
    double *U;
    double *V;
    int *Piv;  // scratch array for pivot values for LU factorization

    void MatrixCopy(double *A, double *B);
    void MatrixScale(double alpha, double *A);
    void MatrixAdd(double alpha, double *A, double beta, double *B);
    void MatrixAdd(double alpha, double *A, double *B);
    void MatrixMultiply(double *A, double *B, double *C);
    void AddScaledEyeMatrix(double c, double *A);

    // Disallow copying by "poisoning" copy constructor and assignment operator,
    // i.e. declare private but provide no implementation.
    ExpMHigham(const ExpMHigham &);  // no implementation
    ExpMHigham & operator=(const ExpMHigham &);  // no implementation
};

#endif
