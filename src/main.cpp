#include "dnn.hpp"

int main(int argc, char const *argv[])
{
	(void)argc;
	(void)argv;
	srand(time(NULL));
	float dump = rand_float(-1, 1);
	(void)dump;

	MATRIX inputs = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
	MATRIX output = {{0}, {1}, {1}, {0}};

	ARCH arch = {2, 2, 1};
	MODEL sum = model_alloc(arch);
	MODEL grad = model_alloc(arch);

	/*sum.weight = rand_mat(1, 1, -1, 1);
	sum.bias = rand_mat(1, 1, -1, 1);
	sum.prediction = {{0}};
	*/
	// MAT_PRINT(sum.bias);
	std::cout << "Cost = " << cost(sum, inputs, output) << "\n";
	for (size_t i = 0; i < EPOCHS; i++)
	{
		finate_diff(sum, grad, inputs, output);
		learn(sum, grad);
	}
	std::cout << "Cost = " << cost(sum, inputs, output) << "\n";
	// MODEL_PRINT(sum);
	for (size_t i = 0; i < 4; i++)
	{
		MATRIX in = row_mat(inputs, i);
		feed_forward(sum, in);
		MAT_PRINT(sum.layers[sum.depth - 1]);
	}

	/* for (size_t i = 0; i < EPOCHS; i++)
	{
		// float dw = finate_diff(input, output, w[0], b[0]);
		float c = cost(input, output, w[0], b[0]);
		float dw = (cost(input, output, w[0] + EPS, b[0]) - c) / EPS;
		float db = (cost(input, output, w[0], b[0] + EPS) - c) / EPS;
		w[0] -= dw * LRATE;
		b[0] -= db * LRATE;
	}

	std::cout << "Cost = " << cost(input, output, w[0], b[0]) << "\n";
	std::cout << "--------------------------------\n";
	std::cout << "W = " << w[0] << "  \tB = " << b[0] << "\n"; */
	return 0;
}

/*
int main(int argc, char const *argv[])
{
	(void)argc;
	(void)argv;
	srand(time(NULL));

	MATRIX m(4, VECTOR(3));
	MATRIX n(3, VECTOR(4));

	rand_mat(m, -1, 1);
	print_mat(m);
	transpose(n, m);
	std::cout << "\n";
	print_mat(n);

	MATRIX a = {{2, 1, 3}, {-2, 2, 1}};
	MATRIX b = {{2, 1}, {3, 2}, {-2, 2}};

	MATRIX r(2, VECTOR(2));

	dot_prod(r, a, b);
	std::cout << "\n";
	print_mat(r);
	return 0;
} */