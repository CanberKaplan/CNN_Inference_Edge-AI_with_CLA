/*
 * convolution.c
 *
 *  Created on: Feb 11, 2026
 *      Author: canberk
 */
#include "convolution.h"

//void convolution_optimized(const float* input, float* output, const float* filters, const float* bias, int in_dim, int out_dim)
//{
//    uint16_t f, d, c, fi, fj;
//    uint32_t filter_offset, output_offset, input_row_ptr;
//
//    for (f = 0; f < NUM_FILTERS; f++) {
//
//        filter_offset = f * (FILTER_SIZE * FILTER_SIZE);
//        output_offset = f * (out_dim * out_dim);
//
//        for (d = 0; d < out_dim; d++) {
//            for (c = 0; c < out_dim; c++) {
//                float sum = 0.0f;
//
//                for (fi = 0; fi < FILTER_SIZE; fi++) {
//
//                    input_row_ptr = (d + fi) * in_dim;
//
//                    for (fj = 0; fj < FILTER_SIZE; fj++) {
//
//                        sum += input[input_row_ptr + (c + fj)] * filters[filter_offset + (fi * FILTER_SIZE) + fj];
//                    }
//                }
//
//                output[output_offset + (d * out_dim) + c] = sum + bias[f];
//            }
//        }
//    }
//}

void convolution(const float* input, float* output, const float* filters, const float* bias, int in_h, int in_w)
{
    uint16_t f, r, c, fi, fj;

    // Calculate output dimensions based on "Valid" convolution rules
    int out_h = in_h - FILTER_SIZE + 1;
    int out_w = in_w - FILTER_SIZE + 1;

    for (f = 0; f < NUM_FILTERS; f++) {

        uint32_t filter_offset = f * (FILTER_SIZE * FILTER_SIZE);
        uint32_t output_offset = f * (out_h * out_w);

        for (r = 0; r < out_h; r++) {
            for (c = 0; c < out_w; c++) {
                float sum = 0.0f;

                for (fi = 0; fi < FILTER_SIZE; fi++) {
                    // Stride across the input is always the full input width
                    uint32_t input_row_ptr = (r + fi) * in_w;

                    for (fj = 0; fj < FILTER_SIZE; fj++) {
                        sum += input[input_row_ptr + (c + fj)] * filters[filter_offset + (fi * FILTER_SIZE) + fj];
                    }
                }

                // Stride across the output is the calculated output width
                output[output_offset + (r * out_w) + c] = sum + bias[f];
            }
        }
    }
}

//void convolution_optimized(const float* input, int16_t* output, const float* filters, const float* bias, int16_t in_dim, int16_t out_dim, float input_scale, float output_scale)
//{
//    uint16_t f, d, c, fi, fj;
//
//
//    for (f = 0; f < NUM_FILTERS; f++) {
//
//        uint32_t filter_offset = f * (FILTER_SIZE * FILTER_SIZE);
//        uint32_t output_offset = f * (out_dim * out_dim);
//
//        for (d = 0; d < out_dim; d++) {
//            for (c = 0; c < out_dim; c++) {
//                float sum = 0.0f;
//
//                for (fi = 0; fi < FILTER_SIZE; fi++) {
//
//                    uint32_t input_row_ptr = (d + fi) * in_dim;
//
//                    for (fj = 0; fj < FILTER_SIZE; fj++) {
//
//                        float val = (float)input[input_row_ptr + (c + fj)] * input_scale;
//
//
//                        sum += val * filters[filter_offset + (fi * FILTER_SIZE) + fj];
//                    }
//                }
//
//                // 3. Add float bias
//                float res = sum + bias[f];
//
//                // 4. Quantize float result back for the next layer
//                float quantized_res = res / output_scale;
//
//                // Saturate to 8-bit range (but keeping in int16_t container)
//                if (quantized_res > 127.0f) quantized_res = 127.0f;
//                if (quantized_res < -128.0f) quantized_res = -128.0f;
//
//                output[output_offset + (d * out_dim) + c] = (int16_t)quantized_res;
//            }
//        }
//    }
//}

void relu_activation(float* data, uint32_t size)
{
    uint32_t i;

    for (i = 0; i < size; i++) {

        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
}

//void relu_activation(int16_t* data, uint32_t size)
//{
//    uint32_t i;
//    for (i = 0; i < size; i++) {
//        if (data[i] < 0) {
//            data[i] = 0;
//        }
//    }
//}

//void max_pooling_inplace(float* data, uint16_t input_dim, uint16_t num_filters) {
//    uint16_t f, r, c, i, j;
//    //uint16_t output_dim = input_dim / 2;
//    uint32_t write_idx = 0;
//
//    for (f = 0; f < num_filters; f++) {
//        uint32_t filter_offset = f * (input_dim * input_dim);
//
//        for (r = 0; r < input_dim; r += 2) {
//            for (c = 0; c < input_dim; c += 2) {
//                float max_val = -1e37f;
//
//                for (i = 0; i < 2; i++) {
//                    for (j = 0; j < 2; j++) {
//                        float val = data[filter_offset + (r + i) * input_dim + (c + j)];
//                        if (val > max_val) {
//                            max_val = val;
//                        }
//                    }
//                }
//                data[write_idx++] = max_val;
//            }
//        }
//    }
//
//}
void max_pooling(float* data, uint16_t in_h, uint16_t in_w, uint16_t num_filters) {
    uint16_t f, r, c, i, j;
    uint32_t write_idx = 0;

    // Calculate output dimensions (downsampling by factor of 2)

    for (f = 0; f < num_filters; f++) {
        // Offset to the start of the current feature map in the input data
        // Use the original dimensions (in_h * in_w) to find each filter's start
        uint32_t filter_offset = (uint32_t)f * in_h * in_w;

        for (r = 0; r < in_h - 1; r += 2) {     // Step by 2 rows
            for (c = 0; c < in_w - 1; c += 2) { // Step by 2 columns
                float max_val = -1e37f; // Smallest possible float

                // Look at the 2x2 window
                for (i = 0; i < 2; i++) {
                    uint32_t row_ptr = filter_offset + (r + i) * in_w;
                    for (j = 0; j < 2; j++) {
                        float val = data[row_ptr + (c + j)];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }

                // Write the winner to the front of the array
                data[write_idx++] = max_val;
            }
        }
    }
}

//void max_pooling_inplace(int16_t* data, uint16_t input_dim, uint16_t num_filters) {
//    uint16_t f, r, c, i, j;
//    uint32_t write_idx = 0;
//
//    for (f = 0; f < num_filters; f++) {
//        uint32_t filter_offset = f * (input_dim * input_dim);
//        for (r = 0; r < input_dim; r += 2) {
//            for (c = 0; c < input_dim; c += 2) {
//                int16_t max_val = -32768;
//
//                for (i = 0; i < 2; i++) {
//                    for (j = 0; j < 2; j++) {
//                        int16_t val = data[filter_offset + (r + i) * input_dim + (c + j)];
//                        if (val > max_val) {
//                            max_val = val;
//                        }
//                    }
//                }
//                data[write_idx++] = max_val;
//            }
//        }
//    }
//}



//void convolution_layer_2(const float* input, float* output, const float* filters, const float* bias) {
//    uint16_t out_f, in_c, r, c, fi, fj;
//    const uint16_t in_dim = 13;
//    const uint16_t out_dim = 11;
//
//    for (out_f = 0; out_f < NUM_FILTERS; out_f++) {
//        for (r = 0; r < out_dim; r++) {
//            for (c = 0; c < out_dim; c++) {
//                float sum = 0.0f;
//
//                for (in_c = 0; in_c < NUM_FILTERS; in_c++) {
//                    uint32_t in_offset = in_c * (in_dim * in_dim);
//                    uint32_t filter_offset = (out_f * NUM_FILTERS * 9) + (in_c * 9);
//
//                    for (fi = 0; fi < 3; fi++) {
//                        for (fj = 0; fj < 3; fj++) {
//                            sum += input[in_offset + (r + fi) * in_dim + (c + fj)] * filters[filter_offset + fi * 3 + fj];
//                        }
//                    }
//                }
//                output[out_f * (out_dim * out_dim) + r * out_dim + c] = sum + bias[out_f];
//            }
//        }
//    }
//}
void convolution_2(const float* input, float* output, const float* filters, const float* bias, uint16_t in_h, uint16_t in_w)
{
    uint16_t out_f, in_c, r, c, fi, fj;

    // Calculate output dimensions (Valid convolution, 3x3 filter)
    uint16_t out_h = in_h - 3 + 1;
    uint16_t out_w = in_w - 3 + 1;

    for (out_f = 0; out_f < NUM_FILTERS; out_f++) {
        for (r = 0; r < out_h; r++) {
            for (c = 0; c < out_w; c++) {
                float sum = 0.0f;

                for (in_c = 0; in_c < NUM_FILTERS; in_c++) {
                    // Use in_w for the row jump in the input
                    uint32_t in_offset = (uint32_t)in_c * (in_h * in_w);

                    // Filter memory layout: (Output Filter index * Total Input Channels * 9)
                    uint32_t filter_offset = (out_f * NUM_FILTERS * 9) + (in_c * 9);

                    for (fi = 0; fi < 3; fi++) {
                        // Point to the correct row using the input width (in_w)
                        uint32_t input_row_ptr = in_offset + (r + fi) * in_w;

                        for (fj = 0; fj < 3; fj++) {
                            sum += input[input_row_ptr + (c + fj)] * filters[filter_offset + fi * 3 + fj];
                        }
                    }
                }

                // Use out_w for the row jump in the output
                output[out_f * (out_h * out_w) + r * out_w + c] = sum + bias[out_f];
            }
        }
    }
}

//void convolution_layer_2(const int16_t* input, int16_t* output, const float* filters,
//                         const float* bias, float input_scale, float output_scale) {
//    uint16_t out_f, in_c, r, c, fi, fj;
//    const uint16_t in_dim = 13;
//    const uint16_t out_dim = 11;
//    const uint16_t filter_sz = 3;
//
//    for (out_f = 0; out_f < 16; out_f++) {
//        for (r = 0; r < out_dim; r++) {
//            for (c = 0; c < out_dim; c++) {
//                float sum = 0.0f;
//
//                for (in_c = 0; in_c < 16; in_c++) {
//                    uint32_t in_offset = in_c * (in_dim * in_dim);
//                    uint32_t filter_offset = (out_f * 16 * 9) + (in_c * 9);
//
//                    for (fi = 0; fi < filter_sz; fi++) {
//                        for (fj = 0; fj < filter_sz; fj++) {
//
//                            float val = (float)input[in_offset + (r + fi) * in_dim + (c + fj)] * input_scale;
//                            sum += val * filters[filter_offset + fi * filter_sz + fj];
//                        }
//                    }
//                }
//
//
//                float res = (sum + bias[out_f]) / output_scale;
//
//
//                if (res > 127.0f) res = 127.0f;
//                if (res < -128.0f) res = -128.0f;
//
//                output[out_f * (out_dim * out_dim) + r * out_dim + c] = (int16_t)res;
//            }
//        }
//    }
//}

//void max_pooling_v2(float* input, float* output, uint16_t input_dim, uint16_t num_filters) {
//    uint16_t f, r, c, i, j;
//    uint16_t output_dim = input_dim / 2;
//    uint32_t write_idx = 0;
//
//    for (f = 0; f < num_filters; f++) {
//        uint32_t input_filter_offset = f * (input_dim * input_dim);
//
//        for (r = 0; r < output_dim; r++) {
//            for (c = 0; c < output_dim; c++) {
//                float max_val = -1e37f;
//
//                for (i = 0; i < 2; i++) {
//                    for (j = 0; j < 2; j++) {
//
//                        uint32_t idx = input_filter_offset + ((r * 2) + i) * input_dim + ((c * 2) + j);
//                        float val = input[idx];
//                        if (val > max_val) {
//                            max_val = val;
//                        }
//                    }
//                }
//
//                output[write_idx++] = max_val;
//            }
//        }
//    }
//}
void max_pooling_2(float* input, float* output, uint16_t in_h, uint16_t in_w, uint16_t num_filters)
{
    uint16_t f, r, c, i, j;

    // Calculate separate output dimensions
    uint16_t out_h = in_h / 2;
    uint16_t out_w = in_w / 2;
    uint32_t write_idx = 0;

    for (f = 0; f < num_filters; f++) {
        // Use full input area (h * w) to jump to the next filter
        uint32_t input_filter_offset = (uint32_t)f * in_h * in_w;

        for (r = 0; r < out_h; r++) {
            for (c = 0; c < out_w; c++) {
                float max_val = -1e37f;

                for (i = 0; i < 2; i++) {
                    // Critical: Jump by the input width (in_w)
                    uint32_t row_offset = input_filter_offset + ((r * 2) + i) * in_w;

                    for (j = 0; j < 2; j++) {
                        // Horizontal position: column index * 2 + window offset
                        float val = input[row_offset + (c * 2) + j];
                        if (val > max_val) {
                            max_val = val;
                        }
                    }
                }
                // Write to the output array sequentially
                output[write_idx++] = max_val;
            }
        }
    }
}

//void max_pooling_v2(int16_t* input, int16_t* output, uint16_t input_dim, uint16_t num_filters) {
//    uint16_t f, r, c, i, j;
//    uint16_t output_dim = input_dim / 2;
//    uint32_t write_idx = 0;
//
//    for (f = 0; f < num_filters; f++) {
//        uint32_t input_filter_offset = f * (input_dim * input_dim);
//
//        for (r = 0; r < output_dim; r++) {
//            for (c = 0; c < output_dim; c++) {
//
//                int16_t max_val = -32768;
//
//                for (i = 0; i < 2; i++) {
//                    for (j = 0; j < 2; j++) {
//
//                        uint32_t idx = input_filter_offset + ((r * 2) + i) * input_dim + ((c * 2) + j);
//                        int16_t val = input[idx];
//
//                        if (val > max_val) {
//                            max_val = val;
//                        }
//                    }
//                }
//
//                output[write_idx++] = max_val;
//            }
//        }
//    }
//}

