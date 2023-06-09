#include "stb_image.hpp"
#include "dnn.hpp"
#include <string.h>

int x, y, n;
unsigned char *get_image_data(const char *filepath)
{
	unsigned char *data = stbi_load(filepath, &x, &y, &n, 0);
	if (data == nullptr or x <= 0 or y <= 0)
	{
		std::cout << "ERROR: could not read image\n";
	}
	return data;
}

int main(int argc, char const *argv[])
{
	(void)argc;
	(void)argv;

	srand(time(NULL));
	float dump = rand_float(-1, 1);
	(void)dump;

	unsigned char *data0 = get_image_data("./testing/0/3.png");
	VECTOR v0 = {};
	for (size_t i = 0; i < (size_t)(x * y); i++)
	{
		v0.push_back(static_cast<float>(data0[i]));
	}
	stbi_image_free(data0);
	// print_vec(v);

	unsigned char *data1 = get_image_data("./testing/1/2.png");
	VECTOR v1 = {};
	for (size_t i = 0; i < (size_t)(x * y); i++)
	{
		v1.push_back(static_cast<float>(data1[i]));
	}
	stbi_image_free(data1);

	MATRIX inputs = {{0, 0}, {1, 0}, {0, 1}, {1, 1}};
	MATRIX output = {{0}, {1}, {1}, {1}};
	/* MATRIX inputs = {{v0}, {v1}};
	MATRIX output = {{0}, {1}};
 */
	ARCH arch = {2, 2, 1};
	MODEL image = model_alloc(arch);
	MODEL grad = model_alloc(arch);

	for (size_t k = 0; k < image.depth; k++)
	{
		MAT_DIM(image.weights[k]);
		MAT_DIM(image.biases[k]);
	}

	std::cout << "Cost = " << cost(image, inputs, output) << "\n";
	for (size_t i = 0; i < EPOCHS; i++)
	{
		// finate_diff(image, grad, inputs, output);
		back_propagation(image, grad, inputs, output);
		learn(image, grad);
		// std::cout << "Iteration " << i << "\n";
	}
	std::cout << "Cost = " << cost(image, inputs, output) << "\n";
	for (size_t i = 0; i < 4; i++)
	{
		MATRIX in = row_mat(inputs, i);
		feed_forward(image, in);
		MATRIX f = MODEL_OUT(image);
		MAT_PRINT(f);
	}
	return 0;
}