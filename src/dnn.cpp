#include "dnn.hpp"

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

float cost(VECTOR input, VECTOR target, float w, float b)
{
	assert(input.size() == target.size());
	float cost = 0;
	size_t n = target.size();
	for (size_t i = 0; i < n; i++)
	{
		float yp = sigmoid(input[i] * w + b);
		float d = yp - target[i];
		cost += d * d;
	}
	return cost / n;
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