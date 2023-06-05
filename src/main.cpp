#include "dnn.hpp"

int main(int argc, char const *argv[])
{
	(void)argc;
	(void)argv;
	srand(time(NULL));

	VECTOR input = {1, 2, 3};
	VECTOR output = {2, 4, 6};

	VECTOR w = rand_vec(1, -1, 1);
	VECTOR b = rand_vec(1, -1, 1);
	std::cout << "W = " << w[0] << "  \tB = " << b[0] << "\n";

	for (size_t i = 0; i < EPOCHS; i++)
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
	std::cout << "W = " << w[0] << "  \tB = " << b[0] << "\n";
	return 0;
}