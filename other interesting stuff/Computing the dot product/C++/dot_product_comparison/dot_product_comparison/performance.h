#pragma once

#include "dot_product.h"
#include <vector>
#include <chrono>
#include <random>
#include <iostream>
#include <algorithm>
#include <numeric>

std::vector<double> createRandomData(int n) {
	std::vector<double> vec(n,0.0);
	// obtain a seed from the system clock:
	unsigned seed = static_cast<int> (std::chrono::system_clock::now().time_since_epoch().count());

	// seeds the random number engine, the mersenne_twister_engine
	std::mt19937 generator(seed);

	// set a distribution range (1 - 100)
	std::uniform_int_distribution<int> distribution(0, 10);


	for (auto& x: vec) {
		x = distribution(generator);
	}
	return vec;
}

void testIterative(size_t n, size_t iter) {
	std::vector<double> x=createRandomData(n);
	std::vector<double> y = createRandomData(n);

	auto rng = std::default_random_engine{};
	// Record start time
	auto start = std::chrono::high_resolution_clock::now();


	// Portion of code to be timed
	for (size_t i = 0; i < iter; i++) {
		dot_product_iterative(x.begin(), x.end(), y.begin());
		//std::inner_product(x.begin(), x.end(), y.begin(),0.0);
		std::shuffle(std::begin(x), std::end(x), rng);
	}

	// Record end time
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "dotproduct_iterative time: " << elapsed.count() << "\n";	

}

void testRecursive(size_t n, size_t iter) {
	std::vector<double> x = createRandomData(n);
	std::vector<double> y = createRandomData(n);
	auto rng = std::default_random_engine{};

	// Record start time
	auto start = std::chrono::high_resolution_clock::now();

	// Portion of code to be timed
	for (size_t i = 0; i < iter; i++) {
		dot_product_recursive(x.begin(), y.begin(), x.size());
		std::shuffle(std::begin(x), std::end(x), rng);
	}

	// Record end time
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "dotproduct_recursive time: " << elapsed.count() << "\n";
}

template<size_t N>
void testTemplate(size_t iter){
	std::vector<double> x = createRandomData(N);
	std::vector<double> y = createRandomData(N);
	auto rng = std::default_random_engine{};

	// Record start time
	auto start = std::chrono::high_resolution_clock::now();

	// Portion of code to be timed
	for (size_t i = 0; i < iter; i++) {
		dot_product_template<N>(x.begin(), y.begin());
		std::shuffle(std::begin(x), std::end(x), rng);
	}

	// Record end time
	auto finish = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = finish - start;
	std::cout << "dotproduct_template time: " << elapsed.count() << "\n";
}
