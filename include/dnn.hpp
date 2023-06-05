#ifndef DNN_H
#define DNN_H

#include <iostream>
#include <vector>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define VECTOR std::vector<float>
#define EPOCHS 1000 * 1000
#define LRATE 1e-1
#define EPS 1e-3

float rand_float(float LO, float HI);
VECTOR rand_vec(size_t size, float LO, float HI);
float cost(VECTOR input, VECTOR target, float w, float b);
// float finate_diff(VECTOR input, VECTOR target, float w);
float sigmoid(float x);

#endif