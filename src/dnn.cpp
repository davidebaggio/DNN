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

MATRIX vec_to_mat(VECTOR &v)
{
	MATRIX m(1, VECTOR(v.size()));
	for (size_t i = 0; i < v.size(); i++)
	{
		m[0][i] = v[i];
	}
	return m;
}

MATRIX row_mat(MATRIX &m, size_t row)
{
	MATRIX n(1, VECTOR(COLS(m)));
	for (size_t i = 0; i < COLS(m); i++)
	{
		n[0][i] = m[row][i];
	}
	return n;
}

void copy_mat(MATRIX &dest, MATRIX &src)
{
	assert(VALIDATESIZE(dest, src));
	for (size_t i = 0; i < ROWS(dest); i++)
	{
		for (size_t j = 0; j < COLS(dest); j++)
		{
			dest[i][j] = src[i][j];
		}
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
	std::cout << "]\n";
}

void print_mat_dim(const MATRIX &m, const char *name)
{
	std::cout << name << ": ROWS = " << ROWS(m) << ", COLS = " << COLS(m) << "\n";
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

float cost(MODEL &m, MATRIX &input, MATRIX &output)
{
	float cost = 0;
	size_t inputs = ROWS(input);
	size_t cols = COLS(m.layers[m.depth - 1]);

	for (size_t i = 0; i < inputs; i++)
	{
		MATRIX in = row_mat(input, i);
		MATRIX out = row_mat(output, i);
		feed_forward(m, in);
		assert(VALIDATESIZE(m.layers[m.depth - 1], out));
		for (size_t j = 0; j < cols; j++)
		{
			float d = m.layers[m.depth - 1][0][j] - out[0][j];
			cost += d * d;
		}
	}
	return cost / inputs;
}

void feed_forward(MODEL &m, MATRIX &input)
{
	copy_mat(m.layers[0], input);
	for (size_t k = 1; k < m.depth; k++)
	{
		dot_prod(m.layers[k], m.layers[k - 1], m.weights[k]);
		matrix_sum(m.layers[k], m.biases[k]);
		mat_sig(m.layers[k]);
	}
}

void finate_diff(MODEL &m, MODEL &g, MATRIX &input, MATRIX &output)
{
	float saved;
	float c = cost(m, input, output);

	for (size_t k = 0; k < m.depth; k++)
	{
		for (size_t i = 0; i < ROWS(m.weights[k]); i++)
		{
			for (size_t j = 0; j < COLS(m.weights[k]); j++)
			{
				saved = m.weights[k][i][j];
				m.weights[k][i][j] += EPS;
				g.weights[k][i][j] = (cost(m, input, output) - c) / EPS;
				m.weights[k][i][j] = saved;
			}
		}

		for (size_t i = 0; i < ROWS(m.biases[k]); i++)
		{
			for (size_t j = 0; j < COLS(m.biases[k]); j++)
			{
				saved = m.biases[k][i][j];
				m.biases[k][i][j] += EPS;
				g.biases[k][i][j] = (cost(m, input, output) - c) / EPS;
				m.biases[k][i][j] = saved;
			}
		}
	}
}

void back_propagation(MODEL &m, MODEL &g, MATRIX &input, MATRIX &output)
{
	assert(ROWS(input) == ROWS(output));
	size_t inputs = ROWS(input);
	size_t outputs = COLS(output);
	for (size_t i = 0; i < inputs; i++)
	{
		MATRIX in = row_mat(input, i);
		feed_forward(m, in);
		for (size_t j = 0; j < outputs; j++)
		{
			g.layers[g.depth - 1][0][j] = MODEL_OUT(m)[0][j] - output[i][j];
		}
		for (size_t l = m.depth - 1; l > 0; l--)
		{
			for (size_t j = 0; j < COLS(m.layers[l]); j++)
			{
				float a = m.layers[l][0][j];
				float da = g.layers[l][0][j];
				g.biases[l][0][j] += a * (1 - a) * 2 * da;
				for (size_t k = 0; k < COLS(m.layers[l - 1]); k++)
				{
					float pa = m.layers[l - 1][0][k];
					float w = m.weights[l][k][j];
					g.weights[l][k][j] += pa * a * (1 - a) * 2 * da;
					g.layers[l - 1][0][k] += w * a * (1 - a) * 2 * da;
					MODEL_PRINT(g);
				}
			}
		}
	}
	for (size_t i = 0; i < g.depth; i++)
	{
		for (size_t j = 0; j < ROWS(g.weights[i]); j++)
		{
			for (size_t k = 0; k < COLS(g.weights[i]); k++)
			{
				g.weights[i][j][k] /= inputs;
			}
		}
		for (size_t j = 0; j < ROWS(g.biases[i]); j++)
		{
			for (size_t k = 0; k < COLS(g.biases[i]); k++)
			{
				g.biases[i][j][k] /= inputs;
			}
		}
	}
}

void learn(MODEL &m, MODEL &g)
{
	for (size_t k = 0; k < m.depth; k++)
	{
		for (size_t i = 0; i < ROWS(m.weights[k]); i++)
		{
			for (size_t j = 0; j < COLS(m.weights[k]); j++)
			{
				m.weights[k][i][j] -= LRATE * g.weights[k][i][j];
			}
		}

		for (size_t i = 0; i < ROWS(m.biases[k]); i++)
		{
			for (size_t j = 0; j < COLS(m.biases[k]); j++)
			{
				m.biases[k][i][j] -= LRATE * g.biases[k][i][j];
			}
		}
	}
}

float sigmoid(float x)
{
	return (1 / (1 + expf(-x)));
}

void mat_sig(MATRIX &m)
{
	for (size_t i = 0; i < ROWS(m); i++)
	{
		for (size_t j = 0; j < COLS(m); j++)
		{
			m[i][j] = sigmoid(m[i][j]);
		}
	}
}

// Model functions

MODEL model_alloc(ARCH arch)
{
	MODEL m;
	m.depth = arch.size();
	MATRIX i(1, VECTOR(arch[0]));
	m.layers.push_back(i);
	for (size_t i = 0; i < arch.size() - 1; i++)
	{
		MATRIX w = rand_mat(arch[i], arch[i + 1], -1, 1);
		m.weights.push_back(w);

		MATRIX b = rand_mat(1, arch[i + 1], -1, 1);
		m.biases.push_back(b);

		MATRIX h(1, VECTOR(arch[i + 1]));
		m.layers.push_back(h);
	}
	return m;
}

void print_model(const MODEL &m, const char *name)
{
	std::cout << name << ":\n";
	for (size_t k = 0; k < m.depth; k++)
	{
		MAT_PRINT(m.weights[k]);
		MAT_PRINT(m.biases[k]);
	}
}
