#ifndef __IntegratorStats__
#define __IntegratorStats__

class KrylovStats
{
public:
    long numProjections;
    long numIterations;
    long maxIterations;
    long minIterations;
    long numRejections;
    long minRejections;
    long maxRejections;
    long numMatrixExponentials;

    KrylovStats();
    void NewProjection(long iterations, long rejections);
};

class IntegratorStats
{
public:
    double cpuTime;
    double error;
    long numTimeSteps;
    int numTerms;
    KrylovStats *krylovStats;

    IntegratorStats(int projections);
    ~IntegratorStats();
    void Step();
    void PrintStats();
    bool WriteStats(const char filename[]);
    //void MatlabPrintStats();
};

#endif
