#include "dnn.hpp"

int main(int argc, char const *argv[])
{
	(void)argc;
	(void)argv;
	srand(time(NULL));
	float dump = rand_float(-1, 1);
	(void)dump;

	MATRIX input = {{1}, {2}, {3}};
	MATRIX output = {{2}, {4}, {6}};
	MATRIX weight = rand_mat(3, 1, -1, 1);
	MATRIX bias = rand_mat(1, 1, -1, 1);

	MAT_PRINT(weight);
	std::cout << "\n";
	MAT_PRINT(bias);
	std::cout << "\n";

	MATRIX prediction(1, VECTOR(1));
	feed_forward(prediction, input, weight, bias);
	float c = cost(prediction, output);

	std::cout << "Cost = " << c;
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