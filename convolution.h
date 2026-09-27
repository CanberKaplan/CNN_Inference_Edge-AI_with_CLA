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


// ==========================================================================
// EKLENDI (asama bazli CLA/CPU bolusumu): dense_layer.cla icindeki
// conv1_pool_fused_cla / conv2_pool_fused_cla ve Cla1Task'taki FC dongusunun
// CPU IKIZLERI. Matematik BIREBIR ayni (ayni olcekler, ayni fused pooling,
// ayni bias-olcek tasimasi) -- bir asamayi cla_pipeline_config.h'tan CPU'ya
// almak SONUCU degil, sadece nerede hesaplandigini degistirir.
//
// Yukaridaki eski (fused OLMAYAN, tek scale_factor'lu) convolution_int16 /
// max_pooling_* fonksiyonlari MNIST prototipinden kalma ve artik bu boru
// hattinda kullanilmiyor; bias'i kendi olcegiyle tasimadiklari icin bu
// modelde YANLIS sonuc verirler. Yeni kod asagidakileri kullanmali.
// ==========================================================================

// conv1 + relu + pool1, tek gecis: ara (havuzlanmamis) conv ciktisi
// tutulmaz. input: IMAGE_H x IMAGE_W, pooled_out: NUM_FILTERS x
// POOL1_OUT_H x POOL1_OUT_W.
void conv1_pool_fused_cpu(const int16_t* input, int16_t* pooled_out,
                           const int16_t* filters, const int16_t* bias);

// conv2 + relu + pool2, tek gecis. input: pool1 ciktisi, pooled_out:
// NUM_FILTERS x POOL2_OUT_H x POOL2_OUT_W (= dense girdisi).
void conv2_pool_fused_cpu(const int16_t* input, int16_t* pooled_out,
                           const int16_t* filters, const int16_t* bias);

// dense/FC: input DENSE_LAYER_INPUT uzunlugunda, weights satir-bazli
// (sinif k'nin agirliklari weights[k*DENSE_LAYER_INPUT ...]), output
// NUM_CLASSES adet float logit.
void dense_fc_cpu(const int16_t* input, const int16_t* weights,
                   const int16_t* bias, float* output);

#endif /* CONVOLUTION_H_ */
