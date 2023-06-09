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
#define VALIDATESIZE(m, n) ROWS((m)) == ROWS((n)) && COLS((m)) == COLS((n))
#define VALIDATEDOT(m, n) COLS((m)) == ROWS((n))
#define VALIDATETRANS(m, n) VALIDATEDOT(m, n) && ROWS((m)) == COLS((n))
#define MAT_PRINT(m) print_mat((m), #m)
#define MAT_DIM(m) print_mat_dim((m), #m)
#define MODEL_PRINT(m) print_model((m), #m)
#define MODEL_OUT(m) (m).layers[(m).depth - 1]
#define ARCH std::vector<int>

#define EPOCHS 100
#define LRATE 1e-2
#define EPS 1e-1

typedef struct
{
	size_t depth;
	std::vector<MATRIX> weights;
	std::vector<MATRIX> biases;
	std::vector<MATRIX> layers;
} MODEL;

// define vector & matrix
float rand_float(float LO, float HI);
VECTOR rand_vec(size_t size, float LO, float HI);
void rand_vec(VECTOR &v, float LO, float HI);
void fill_vec(VECTOR &v, float value);
MATRIX rand_mat(size_t row, size_t col, float LO, float HI);
void rand_mat(MATRIX &m, float LO, float HI);
void fill_mat(MATRIX &m, float value);
MATRIX vec_to_mat(VECTOR &v);
MATRIX row_mat(MATRIX &v, size_t row);
void copy_mat(MATRIX &dest, MATRIX &src);

// print vector & matrix
void print_vec(const VECTOR &v);
void print_mat(const MATRIX &m, const char *name);
void print_mat_dim(const MATRIX &m, const char *name);

// matrix operation
void scalar_prod(MATRIX &dest, MATRIX &m, float scalar);
void matrix_sum(MATRIX &m, MATRIX &n);
void transpose(MATRIX &dest, MATRIX &m);
void dot_prod(MATRIX &dest, MATRIX &m, MATRIX &n);

// DNN functions
float cost(MODEL &m, MATRIX &in, MATRIX &out);
void feed_forward(MODEL &m, MATRIX &input);
void finate_diff(MODEL &m, MODEL &g, MATRIX &input, MATRIX &output);
void back_propagation(MODEL &m, MODEL &g, MATRIX &input, MATRIX &output);
void learn(MODEL &m, MODEL &g);
float sigmoid(float x);
void mat_sig(MATRIX &m);

// model function
MODEL model_alloc(ARCH arch);
void print_model(const MODEL &m, const char *name);

#endif