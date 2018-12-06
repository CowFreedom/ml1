#include <iostream>
#include <array>
#include "dot_product.h"
#include "performance.h"

int main() {

	size_t iterations = 100000;
	constexpr size_t vectorlength = 2*2048;
	testIterative(vectorlength, iterations);
	testRecursive(vectorlength, iterations);
	testTemplate<vectorlength>(iterations);
	std::cin.get();
}