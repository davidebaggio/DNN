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

	std::cout << "Cost = " << cost(sum, inputs, output) << "\n";
	for (size_t i = 0; i < EPOCHS; i++)
	{
		finate_diff(sum, grad, inputs, output);
		learn(sum, grad);
	}
	std::cout << "Cost = " << cost(sum, inputs, output) << "\n";
	for (size_t i = 0; i < 4; i++)
	{
		MATRIX in = row_mat(inputs, i);
		feed_forward(sum, in);
		MAT_PRINT(sum.layers[sum.depth - 1]);
	}
	return 0;
}