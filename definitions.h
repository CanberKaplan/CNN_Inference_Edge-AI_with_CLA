/*
 * definitions.h
 *
 *  Created on: Feb 11, 2026
 *      Author: canbe
 */

#ifndef DEFINITIONS_H_
#define DEFINITIONS_H_


#define IMAGE_SIZE 28
#define NUM_FILTERS 17
#define FILTER_SIZE 3  //3X3
#define POLLING_SIZE 2 //2X2
#define CONV_OUT_SIZE (IMAGE_SIZE - POLLING_SIZE)
#define MAX_POLL_OUT_ELEMENT (CONV_OUT_SIZE/2) * (CONV_OUT_SIZE/2) * NUM_FILTERS
#define CONV2_IN_SIZE (CONV_OUT_SIZE/POLLING_SIZE)
#define CONV2_OUT_SIZE  (CONV2_IN_SIZE - POLLING_SIZE)
#define MAX_ELEMENTS 11492 // 16 * 26 * 26
#define DENSE_LAYER_INPUT 425 // Model weights/10. division is to fit into CLA memory




#endif /* DEFINITIONS_H_ */
