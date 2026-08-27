/*
 * convolution.h
 *
 *  Created on: Feb 11, 2026
 *      Author: canberk
 */

#include <stdlib.h>
#include <stdint.h>
#include "definitions.h"

#ifndef CONVOLUTION_H_
#define CONVOLUTION_H_

void convolution(const float* input, float* output, const float* filters, const float* bias, int in_h, int in_w);

void relu_activation(float* data, uint32_t size);

void max_pooling(float* data, uint16_t in_h, uint16_t in_w, uint16_t num_filters);

void convolution_2(const float* input, float* output, const float* filters, const float* bias, uint16_t in_h, uint16_t in_w);

void max_pooling_2(float* input, float* output, uint16_t in_h, uint16_t in_w, uint16_t num_filters);


void convolution_int16(const int16_t* input, int16_t* output, const int16_t* filters, const int16_t* bias, int in_h, int in_w, float scale_factor);

void relu_activation_int16(int16_t* data, uint32_t size);

void max_pooling_int16(int16_t* data, uint16_t in_h, uint16_t in_w, uint16_t num_filters);

void max_pooling_2x1_int16(const int16_t* input, int16_t* output,
                                 uint16_t in_h, uint16_t in_w, uint16_t num_filters);

void convolution_2_int16(const int16_t* input, int16_t* output, const int16_t* filters, const int16_t* bias, uint16_t in_h, uint16_t in_w, float scale_factor);

void max_pooling_2_int16(int16_t* input, int16_t* output, uint16_t in_h, uint16_t in_w, uint16_t num_filters);

#endif /* CONVOLUTION_H_ */
