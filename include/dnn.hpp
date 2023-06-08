#ifndef DNN_H
#define DNN_H

#include <iostream>
#include <vector>
#include <assert.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define VECTOR std::vector<float>
#define MATRIX std::vector<std::vector<float>>
#define ROWS(m) (m).size()
#define COLS(m) ((m)[0]).size()
#define VALIDATESIZE(m, n) ROWS(m) == ROWS(n) && COLS(m) == COLS(n)
#define VALIDATEDOT(m, n) COLS(m) == ROWS(n)
#define VALIDATETRANS(m, n) VALIDATEDOT(m, n) && ROWS(m) == COLS(n)
#define MAT_PRINT(m) print_mat((m), #m)

#define EPOCHS 1000 * 1000
#define LRATE 1e-1
#define EPS 1e-3

// define vector & matrix
float rand_float(float LO, float HI);
VECTOR rand_vec(size_t size, float LO, float HI);
void rand_vec(VECTOR &v, float LO, float HI);
void fill_vec(VECTOR &v, float value);
MATRIX rand_mat(size_t row, size_t col, float LO, float HI);
void rand_mat(MATRIX &m, float LO, float HI);
void fill_mat(MATRIX &m, float value);

// print vector & matrix
void print_vec(const VECTOR &v);
void print_mat(const MATRIX &m, const char *name);

// matrix operation
void scalar_prod(MATRIX &dest, MATRIX &m, float scalar);
void matrix_sum(MATRIX &m, MATRIX &n);
void transpose(MATRIX &dest, MATRIX &m);
void dot_prod(MATRIX &dest, MATRIX &m, MATRIX &n);

// DNN functions
float cost(MATRIX predicted, MATRIX target);
void feed_forward(MATRIX &prediction, MATRIX &input, MATRIX &weight, MATRIX &bias);

// float finate_diff(VECTOR input, VECTOR target, float w);
float sigmoid(float x);

#endif