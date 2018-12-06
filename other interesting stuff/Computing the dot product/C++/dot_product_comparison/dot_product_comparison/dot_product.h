#pragma once

#include <array>
#include <iterator>

template<class I>
typename std::enable_if<std::is_same<double, typename std::iterator_traits<I>::value_type>::value, double>::type 
dot_product_iterative(I first1, I last1, I first2) {
	double sum = 0.0;

	while (first1 != last1) {
		sum = std::move(sum) + *first1 * *first2;
		//sum+= *first1 * *first2;
		first1++;
		first2++;
	}
	return sum;
}

template<class I>
typename std::enable_if<std::is_same<double, typename std::iterator_traits<I>::value_type>::value, double>::type
dot_product_recursive(I first1, I first2, size_t n) {
	if (n > 1) {
		return *first1 * *first2 + dot_product_recursive(++first1, ++first2,n-1);
	}
	else {
		return *first1 * *first2;
	}
}

template <size_t N, class T>
class adder {
public:
	static double dot_product(T first, T last) {
		return *first * *last +adder<N-1,T>::dot_product(++first,++last);
	}
};

template <class T>
class adder<1,T> {
public:
	static double dot_product(T first, T last) {
		return *first* *last;
	}
};


template <size_t N, class I>
inline static typename std::enable_if<std::is_same<double, typename std::iterator_traits<I>::value_type>::value, double>::type
dot_product_template(I first1, I first2) {
	return adder<N,I>::dot_product(first1,first2);
}

//typename enable_if<std::is_same<typename iterator_traits<I>::value_type, double>>::type