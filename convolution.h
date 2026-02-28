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
//void convolution_optimized(const float* input, float* output, const float* filters, const float* bias, int in_dim, int out_dim);
//void convolution_optimized(const float* input, int16_t* output,const float* filters, const float* bias, int16_t in_dim, int16_t out_dim,float input_scale, float output_scale);
void relu_activation(float* data, uint32_t size);
//void relu_activation(int16_t* data, uint32_t size);
void max_pooling(float* data, uint16_t in_h, uint16_t in_w, uint16_t num_filters);
//void max_pooling_inplace(float* data, uint16_t input_dim, uint16_t num_filters);
//void max_pooling_inplace(int16_t* data, uint16_t input_dim, uint16_t num_filters);
void convolution_2(const float* input, float* output, const float* filters, const float* bias, uint16_t in_h, uint16_t in_w);
//void convolution_layer_2(const float* input, float* output, const float* filters, const float* bias);
//void convolution_layer_2(const int16_t* input, int16_t* output, const float* filters, const float* bias, float input_scale, float output_scale);
void max_pooling_2(float* input, float* output, uint16_t in_h, uint16_t in_w, uint16_t num_filters);
//void max_pooling_v2(float* input, float* output, uint16_t input_dim, uint16_t num_filters);
//void max_pooling_v2(int16_t* input, int16_t* output, uint16_t input_dim, uint16_t num_filters);
#endif /* CONVOLUTION_H_ */
