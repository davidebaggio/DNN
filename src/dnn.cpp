#include "dnn.hpp"

// define vector & matrix

float rand_float(float LO, float HI)
{
	return ((HI - LO) * ((float)rand() / (float)RAND_MAX)) + LO;
}

VECTOR rand_vec(size_t size, float LO, float HI)
{
	VECTOR v(size);
	for (size_t i = 0; i < size; i++)
	{
		v[i] = rand_float(LO, HI);
	}
	return v;
}

void rand_vec(VECTOR &v, float LO, float HI)
{
	size_t lenght = v.size();
	for (size_t i = 0; i < lenght; i++)
	{
		v[i] = rand_float(LO, HI);
	}
}

void fill_vec(VECTOR &v, float value)
{
	for (size_t i = 0; i < v.size(); i++)
	{
		v[i] = value;
	}
}

MATRIX rand_mat(size_t row, size_t col, float LO, float HI)
{
	MATRIX m(row, VECTOR(col));
	for (size_t i = 0; i < row; i++)
	{
		for (size_t j = 0; j < col; j++)
		{
			m[i][j] = rand_float(LO, HI);
		}
	}
	return m;
}

void rand_mat(MATRIX &m, float LO, float HI)
{
	size_t row = m.size();
	for (size_t i = 0; i < row; i++)
	{
		rand_vec(m[i], LO, HI);
	}
}

void fill_mat(MATRIX &m, float value)
{
	for (size_t i = 0; i < m.size(); i++)
	{
		fill_vec(m[i], value);
	}
}

// print vector & matrix

void print_vec(const VECTOR &v)
{
	for (size_t i = 0; i < v.size(); i++)
	{
		std::cout << v[i] << ", \t ";
	}
	std::cout << "\n";
}

void print_mat(const MATRIX &m, const char *name)
{
	std::cout << name << "[\n";
	for (size_t i = 0; i < m.size(); i++)
	{
		std::cout << "\t";
		print_vec(m[i]);
	}
	std::cout << "]";
}

// matrix operation

void scalar_prod(MATRIX &dest, MATRIX &m, float scalar)
{
	assert(VALIDATESIZE(dest, m));
	for (size_t i = 0; i < ROWS(m); i++)
	{
		for (size_t j = 0; j < COLS(m); j++)
		{
			dest[i][j] = m[i][j] * scalar;
		}
	}
}

void matrix_sum(MATRIX &m, MATRIX &n)
{
	assert(VALIDATESIZE(m, n));
	for (size_t i = 0; i < ROWS(m); i++)
	{
		for (size_t j = 0; j < COLS(m); j++)
		{
			m[i][j] += n[i][j];
		}
	}
}
void transpose(MATRIX &dest, MATRIX &m)
{
	assert(VALIDATETRANS(dest, m));
	for (size_t i = 0; i < ROWS(m); i++)
	{
		for (size_t j = 0; j < COLS(m); j++)
		{
			dest[j][i] = m[i][j];
		}
	}
}

void dot_prod(MATRIX &dest, MATRIX &m, MATRIX &n)
{
	assert(VALIDATEDOT(m, n));
	assert(ROWS(dest) == ROWS(m) && COLS(dest) == COLS(n));

	fill_mat(dest, 0);
	for (size_t i = 0; i < ROWS(m); ++i)
	{
		for (size_t j = 0; j < COLS(n); ++j)
		{
			for (size_t k = 0; k < COLS(m); ++k)
			{
				dest[i][j] += m[i][k] * n[k][j];
			}
		}
	}
}

// DNN functions

float cost(MATRIX predicted, MATRIX target)
{
	assert(VALIDATESIZE(predicted, target));
	float cost = 0;
	size_t rows = ROWS(predicted);
	size_t cols = COLS(predicted);
	for (size_t i = 0; i < rows; i++)
	{
		for (size_t j = 0; j < cols; j++)
		{
			float d = predicted[i][j] - target[i][j];
			cost += d * d;
		}
	}
	return cost / (rows * cols);
}

void feed_forward(MATRIX &prediction, MATRIX &input, MATRIX &weight, MATRIX &bias)
{
	dot_prod(prediction, input, weight);
	matrix_sum(prediction, bias);
}

/* float finate_diff(VECTOR input, VECTOR target, float w)
{
	return 0;
	// return (cost(input, target, w + EPS) - cost(input, target, w)) / EPS;
} */

float sigmoid(float x)
{
	return (1.f / (1.f + expf(-x)));
}
