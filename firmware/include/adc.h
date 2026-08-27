/*
 * adc.h
 *
 *  Created on: Aug 10, 2026
 *      Author: canbe
 */

#ifndef FIRMWARE_INCLUDE_ADC_H_
#define FIRMWARE_INCLUDE_ADC_H_

#include "F28x_Project.h"

//
// Defines
//
#define RESULTS_BUFFER_SIZE 8448    // Buffer for storing conversion results
                                    // (size must be multiple of 16)

//
// Globals
//
extern Uint16 adcData0[RESULTS_BUFFER_SIZE];
extern Uint16 adcData1[RESULTS_BUFFER_SIZE];
extern volatile Uint16 done;

__interrupt void adca1_isr(void);
__interrupt void dmach1_isr(void);
void ConfigureEPWM(void);
void ConfigureADC(void);
void SetupADCContinuous(volatile struct ADC_REGS * adcRegs, Uint16 channel);
void DMAInit(void);


#endif /* FIRMWARE_INCLUDE_ADC_H_ */
