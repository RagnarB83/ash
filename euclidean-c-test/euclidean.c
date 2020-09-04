#include "euclidean.h"
#include <math.h>

double euclidean(double x[128], double y[128])
{
    double Sum;
    for(int i=0;i<128;i++)
    {
        Sum = Sum + pow((x[i]-y[i]),2.0);
    }
    return Sum;
}
