

/**
 * main.c
 */
#include "F28x_Project.h"     // Includes everything: PieCtrlRegs, Cla1Regs, etc.
#include "cla_dense_layer_shared.h"
#include "weights.h"
#include <string.h>
#include <stdint.h>
#include "convolution.h"
#include "definitions.h"

#define INPUT_SIZE 2704

#ifdef __cplusplus
#pragma DATA_SECTION("CpuToCla1MsgRAM");
float fVal; //input
#pragma DATA_SECTION("Cla1ToCpuMsgRAM");
float fResult;  //output
#else
#pragma DATA_SECTION(cla_count_param,"CpuToCla1MsgRAM");
uint16_t cla_count_param; //input

#pragma DATA_SECTION(cla_offset_param,"CpuToCla1MsgRAM");
uint16_t cla_offset_param; //input

#pragma DATA_SECTION(fResult,"Cla1ToCpuMsgRAM");
float fResult;  //output

#pragma DATA_SECTION(input_vector,"CLADataLS0");
//float input_vector[400] = {
//    0.4733, 1.2268, 0.9893, 0.2407, 0.0000, 0.6350, 0.5095, 0.9629, 0.0000, 0.0000,
//    0.0000, 0.7855, 1.3916, 0.0000, 0.0000, 0.0000, 0.1389, 0.8111, 2.0923, 0.0000,
//    0.0000, 1.2783, 1.7907, 1.5964, 0.0000, 1.5361, 4.3063, 4.5195, 3.8731, 1.1817,
//    0.0000, 0.0000, 0.0000, 0.0000, 0.2993, 1.5258, 1.5696, 0.0000, 0.0000, 1.3036,
//    0.1985, 1.2486, 3.8879, 1.4119, 0.0000, 0.7916, 3.3021, 3.5294, 0.0000, 0.0000,
//    0.0000, 0.0000, 0.0000, 0.7786, 0.4688, 0.3675, 1.7620, 0.0000, 2.9293, 1.2972,
//    0.7518, 0.0000, 1.6125, 2.3197, 0.8664, 0.0000, 0.4619, 0.6203, 1.9687, 3.9399,
//    0.0000, 0.0000, 3.2546, 5.8650, 2.6416, 1.3539, 0.3903, 0.0000, 0.0000, 0.0000,
//    3.1005, 1.2432, 0.0000, 0.0000, 0.0000, 0.9369, 4.2158, 2.8721, 0.0000, 0.0000,
//    0.0000, 1.9737, 1.9936, 1.4967, 0.0000, 0.9223, 1.4770, 0.9815, 0.6446, 0.0000,
//
//    0.0000, 1.8599, 2.5631, 2.5440, 1.4628, 1.2577, 3.7673, 1.9454, 2.4025, 0.8688,
//    0.0000, 1.5126, 2.5191, 0.0000, 1.8759, 0.0000, 0.3033, 0.0000, 3.1129, 1.7418,
//    0.0000, 0.7402, 6.2305, 5.9072, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
//    2.5837, 6.0020, 5.9696, 0.5067, 0.2326, 0.0910, 3.6572, 5.2748, 1.4921, 0.0000,
//    0.0000, 1.2283, 4.4499, 4.1235, 1.2306, 0.0000, 0.0000, 2.2027, 3.2167, 1.0202,
//    1.5974, 1.9865, 3.0364, 0.1684, 0.0000, 0.0000, 1.9763, 2.0106, 1.4551, 0.0000,
//    0.0000, 1.7369, 5.3627, 6.7706, 0.0000, 0.0000, 0.0000, 0.4251, 2.1110, 0.0000,
//    0.1714, 1.9333, 0.0000, 0.3054, 0.0000, 1.5193, 4.5945, 5.0298, 3.2719, 0.0000,
//    0.0000, 2.1445, 2.3643, 2.5854, 0.7489, 0.0000, 0.0000, 1.7378, 5.7699, 2.0719,
//    0.0000, 0.1122, 3.0756, 0.9519, 0.9654, 0.3538, 4.1214, 4.1372, 1.0110, 1.1748,
//
//    1.6290, 2.6056, 2.5204, 0.7770, 0.3168, 1.1017, 5.1916, 2.6770, 0.0000, 0.3424,
//    2.1488, 3.0287, 0.3377, 0.0000, 0.0343, 0.3168, 2.5682, 4.4899, 1.7355, 0.0000,
//    0.8965, 2.1001, 0.0000, 1.4255, 0.9996, 0.2892, 1.0492, 1.5599, 4.4057, 2.1776,
//    0.0000, 0.0000, 0.0000, 4.8063, 2.6546, 0.3519, 0.0000, 0.0000, 3.0684, 4.1867,
//    0.3519, 0.3519, 0.0000, 0.0000, 3.8111, 0.3519, 0.0000, 0.0000, 1.2122, 2.2013,
//
//    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
//    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
//    0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5650, 0.0000,
//    0.0000, 0.0000, 0.0000, 3.4536, 0.0000, 0.0000, 0.0000, 0.0000, 1.5312, 1.5425,
//    0.0000, 0.0000, 0.0000, 1.7494, 4.5757, 0.0000, 0.0000, 0.0000, 2.4622, 2.5209,
//
//    0.0000, 0.0000, 0.0000, 0.0000, 0.1236, 1.6903, 4.7599, 0.0000, 1.9051, 0.9555,
//    1.9246, 0.7380, 1.1485, 0.0000, 0.0000, 0.1236, 2.1466, 3.4921, 0.0000, 2.5929,
//    0.1236, 0.0000, 0.0000, 6.7431, 5.0169, 0.0000, 1.6943, 1.1013, 1.1056, 0.0000,
//    0.1779, 1.3348, 3.6066, 1.3145, 0.0000, 0.0000, 0.2175, 3.1212, 0.0000, 0.0000,
//    0.0000, 0.0000, 0.3986, 4.3430, 0.0000, 0.0000, 1.7678, 2.8461, 4.0527, 0.0000,
//    2.6385, 1.6162, 0.3388, 0.0000, 0.0012, 0.6286, 3.1651, 3.6608, 2.7947, 2.1277,
//    0.0012, 1.8075, 0.0000, 3.3098, 0.0000, 0.0012, 0.0012, 3.8369, 2.4810, 3.3974,
//    1.2923, 2.9112, 2.5739, 3.1657, 4.2629, 1.8053, 5.4155, 6.8627, 4.5253, 0.0000,
//    0.0000, 2.8958, 4.5707, 4.7508, 0.7339, 0.0000, 0.1253, 4.4335, 6.9328, 2.9400,
//    0.0000, 0.0000, 0.0000, 3.0284, 3.6162, 0.4445, 3.3222, 2.8537, 0.5684, 0.0000
//};
//float input_vector[400] = {   // for 8
//                           0.0000, 0.9087, 1.3174, 0.5873, 0.3479, 0.0000, 4.0275, 0.7787, 0.0000, 0.4079,
//                           0.0000, 2.9604, 0.0419, 0.9504, 1.5076, 2.0298, 1.4569, 0.1879, 0.0000, 0.0000,
//                           1.0222, 0.0000, 0.0000, 0.0000, 0.0000, 0.3634, 2.1383, 4.2302, 4.6357, 4.3990,
//                           0.3358, 0.5864, 0.0000, 0.5701, 0.0000, 1.7891, 2.2817, 0.0000, 3.5149, 0.0000,
//                           3.7485, 4.6757, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.1985,
//                           0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0598, 4.1694, 1.8028, 0.3000,
//                           0.0000, 0.0566, 0.5440, 2.1818, 4.9704, 0.0000, 1.5652, 0.0007, 5.2397, 2.6619,
//                           0.8994, 1.3830, 3.3814, 4.3244, 0.0000, 0.0000, 0.9841, 0.0846, 0.0000, 0.0000,
//                           0.5643, 4.0982, 0.0000, 0.0000, 0.5428, 0.5228, 4.4145, 0.0000, 0.5670, 1.0297,
//                           2.4066, 1.1950, 0.4203, 0.0000, 0.0000, 4.4486, 0.0000, 0.0000, 0.0000, 0.0000,
//                           0.0000, 0.0000, 2.7163, 2.9903, 1.8187, 0.0000, 2.0869, 5.4552, 2.6006, 1.2972,
//                           0.1228, 0.9100, 0.0604, 6.3688, 4.8476, 2.0919, 4.4100, 3.2293, 2.8446, 0.0000,
//                           1.8028, 0.5196, 2.8155, 0.9325, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
//                           0.1887, 1.5773, 4.6009, 7.5206, 4.9860, 0.3953, 2.1289, 0.0000, 2.3006, 4.3774,
//                           0.0000, 1.2264, 2.0092, 1.5831, 0.6686, 5.5004, 7.2147, 7.0545, 2.4695, 0.5950,
//                           0.0000, 0.6605, 0.9116, 2.6533, 3.4198, 0.0000, 2.5533, 1.2176, 1.2230, 3.1434,
//                           0.0000, 3.5389, 6.4847, 0.0000, 0.5011, 2.5409, 0.8547, 2.3269, 1.6398, 0.0000,
//                           2.6334, 4.7527, 2.5010, 1.2805, 0.0000, 0.0000, 2.9760, 4.5409, 4.6789, 4.2552,
//                           0.0000, 0.6011, 3.3900, 3.0476, 1.7437, 1.7026, 2.7897, 5.6861, 4.5930, 0.4198,
//                           4.2551, 3.0655, 3.8540, 2.2145, 0.5493, 0.0000, 0.7521, 0.0000, 0.9508, 0.0000,
//                           0.4420, 1.6822, 2.4012, 2.0810, 1.8556, 0.3507, 0.0000, 1.4710, 6.1312, 1.8362,
//                           2.0636, 3.7151, 0.3729, 0.0000, 1.3588, 1.9559, 0.0000, 0.0000, 0.0000, 3.0107,
//                           3.7471, 2.3252, 2.8082, 2.6525, 0.5036, 0.3519, 0.2975, 0.9117, 1.9253, 1.8702,
//                           0.3519, 0.0000, 1.7188, 0.0000, 0.0000, 0.3519, 0.0000, 1.8923, 0.0000, 0.5040,
//                           0.0000, 0.0000, 0.0000, 4.1689, 1.7954, 0.0000, 0.0000, 0.0000, 2.4372, 0.8316,
//                           0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
//                           0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
//                           0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
//                           0.0000, 0.0000, 1.3507, 0.0000, 1.3677, 0.0000, 0.0000, 0.9843, 0.0000, 2.1911,
//                           0.0000, 0.0000, 0.0000, 4.1213, 0.6420, 0.0000, 0.0000, 0.0000, 3.2967, 0.0000,
//                           0.1236, 0.0042, 0.0000, 0.0000, 0.0000, 0.1236, 0.0000, 3.9871, 3.5361, 0.0000,
//                           0.1236, 0.0000, 0.0000, 0.0000, 7.7305, 0.0000, 0.0000, 0.0000, 4.1391, 5.8360,
//                           4.3575, 5.7085, 8.8804, 6.8812, 1.2710, 0.0000, 0.0000, 1.7844, 0.3770, 0.0000,
//                           0.0000, 4.2594, 3.7114, 0.0000, 3.5886, 0.0000, 3.6209, 2.0082, 1.7092, 3.7741,
//                           1.9113, 2.9531, 2.4033, 1.7822, 0.0000, 1.5747, 0.0000, 1.0458, 0.0000, 0.0000,
//                           0.0514, 3.7596, 1.7945, 0.0878, 0.1643, 0.4702, 2.4166, 1.6841, 3.7988, 1.5297,
//                           0.8436, 1.4980, 2.5553, 3.6186, 1.7861, 2.7893, 0.3043, 2.5401, 1.5580, 2.2057,
//                           0.0000, 0.9241, 0.0000, 4.0613, 1.6363, 0.0000, 1.7610, 4.9368, 6.3240, 6.5275,
//                           0.0000, 1.5431, 1.2345, 2.2218, 5.2700, 0.0663, 0.4465, 5.7970, 3.9880, 1.2555,
//                           4.2098, 4.4036, 3.4786, 2.0585, 0.0000, 1.0078, 4.3364, 2.8030, 1.2718, 0.1325,
//};

float input_vector[DENSE_LAYER_INPUT];

//#define MAX_ELEMENTS 11492 // 16 * 26 * 26
#pragma DATA_ALIGN(workspace, 8)
float workspace[MAX_ELEMENTS] = {0.0f};


#pragma DATA_SECTION(weight,"CLADataLS1");
float weight [DENSE_LAYER_INPUT];

#pragma DATA_SECTION(weight_1,"CLADataLS1");
float weight_1 [200];

#pragma DATA_SECTION(weight_2,"CLADataLS1");
float weight_2 [200];
#endif //__cplusplus


void CLA_runTest(void);
void CLA_configClaMemory(void);
void CLA_initCpu1Cla1(void);
__interrupt void cla1Isr1();

Uint16 count = 0;
Uint16 val= 12;
float output[10];
unsigned int i,j,k;
float cla_accumulated_sum = 0.0f;
size_t dataSize = sizeof(float);
size_t halfSize = 200 * sizeof(float);
size_t sizePart1 = 100 * sizeof(float);
size_t sizePart2 = 300 * sizeof(float);
int main(void)
{
    InitSysCtrl();
    InitGpio(); // Skipped for this example

    EALLOW;
    GpioCtrlRegs.GPCGMUX1.bit.GPIO67 = 0;
    GpioCtrlRegs.GPCDIR.bit.GPIO67 = 1;
    GpioDataRegs.GPCSET.bit.GPIO67 = 1;
    GpioCtrlRegs.GPCPUD.bit.GPIO67 = 0;
    GpioCtrlRegs.GPCCSEL1.bit.GPIO67 = 1;
    EDIS;

    GPIO_SetupPinOptions(18, GPIO_OUTPUT, GPIO_PUSHPULL);
    GPIO_SetupPinMux(18, GPIO_MUX_CPU1, 0);
    GPIO_SetupPinOptions(19, GPIO_OUTPUT, GPIO_PUSHPULL);
    GPIO_SetupPinMux(19, GPIO_MUX_CPU1, 0);

    DINT;
    // Initialize the PIE control registers to their default state.
    InitPieCtrl();

    // Disable CPU interrupts and clear all CPU interrupt flags:


    IER = 0x0000;
    IFR = 0x0000;

    InitPieVectTable();



    CLA_configClaMemory();
    CLA_initCpu1Cla1();

    EINT;  // Enable Global interrupt INTM
    ERTM;  // Enable Global realtime interrupt DBGM


    for(;;)
     {

        convolution(input_image, workspace, filters1, bias1, IMAGE_SIZE, IMAGE_SIZE);
        relu_activation(workspace, MAX_ELEMENTS);
        max_pooling(workspace, CONV_OUT_SIZE, CONV_OUT_SIZE, NUM_FILTERS);

        convolution_2(workspace, &workspace[MAX_POLL_OUT_ELEMENT], filters2, bias2, CONV2_IN_SIZE, CONV2_IN_SIZE);
        relu_activation(&workspace[MAX_POLL_OUT_ELEMENT], NUM_FILTERS * CONV2_OUT_SIZE * CONV2_OUT_SIZE);
        max_pooling_2(&workspace[MAX_POLL_OUT_ELEMENT], input_vector, CONV2_OUT_SIZE, CONV2_OUT_SIZE, NUM_FILTERS);



        for(i = 0; i < 10; i++)
        {

            GpioDataRegs.GPACLEAR.bit.GPIO18 = 1;
           for(j = 0; j < 425; j++)
           {
               weight[j] = weights_data[(i * 425) + j];


           }
           GpioDataRegs.GPASET.bit.GPIO18 = 1;
                   CLA_runTest();
                  DELAY_US(500);
                   output[i] = fResult;
           }

           GpioDataRegs.GPASET.bit.GPIO18 = 1;



     }

}



void CLA_runTest(void)
{
//    fVal =12;

    Cla1ForceTask1();
//    asm(" RPT #8 || NOP");
//    while(Cla1Regs.MIRUN.bit.INT1 == 1);

#if 0
    Cla1ForceTask2andWait();
    WAITSTEP;

    Cla1ForceTask3andWait();
    WAITSTEP;

    Cla1ForceTask4andWait();
    WAITSTEP;

    Cla1ForceTask5andWait();
    WAITSTEP;

    Cla1ForceTask6andWait();
    WAITSTEP;

    Cla1ForceTask7andWait();
    WAITSTEP;
#endif
}
//
////
// CLA_configClaMemory - Configure the CLA memory
//
void CLA_configClaMemory(void)
{
    extern uint32_t Cla1funcsRunStart, Cla1funcsLoadStart, Cla1funcsLoadSize;
    EALLOW;

#ifdef _FLASH
    //
    // Copy over code from FLASH to RAM
    //
    memcpy((uint32_t *)&Cla1funcsRunStart, (uint32_t *)&Cla1funcsLoadStart,
           (uint32_t)&Cla1funcsLoadSize);
#endif //_FLASH

    //
    // Initialize and wait for CLA1ToCPUMsgRAM
    //
    MemCfgRegs.MSGxINIT.bit.INIT_CLA1TOCPU = 1;
    while(MemCfgRegs.MSGxINITDONE.bit.INITDONE_CLA1TOCPU != 1){};

    //
    // Initialize and wait for CPUToCLA1MsgRAM
    //
    MemCfgRegs.MSGxINIT.bit.INIT_CPUTOCLA1 = 1;
    while(MemCfgRegs.MSGxINITDONE.bit.INITDONE_CPUTOCLA1 != 1){};

    //
    // Select LS4RAM and LS5RAM to be the programming space for the CLA
    // First configure the CLA to be the master for LS4 and LS5 and then
    // set the space to be a program block
    //
    MemCfgRegs.LSxMSEL.bit.MSEL_LS4 = 1;
    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS4 = 1; // program memory
    MemCfgRegs.LSxMSEL.bit.MSEL_LS5 = 1;
    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS5 = 1;

    //
    //Next configure LS0RAM and LS1RAM as data spaces for the CLA
    // First configure the CLA to be the master for LS0(1) and then
    // set the spaces to be code blocks
    //
    MemCfgRegs.LSxMSEL.bit.MSEL_LS0 = 1;
    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS0 = 0; // data memory


    MemCfgRegs.LSxMSEL.bit.MSEL_LS1 = 1;
    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS1 = 0;


    MemCfgRegs.LSxMSEL.bit.MSEL_LS2 = 1;
    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS2 = 0;
    EDIS;
}

//
// CLA_initCpu1Cla1 - Initialize CLA1 task vectors and end of task interrupts
//
void CLA_initCpu1Cla1(void)
{
    //
    // Compute all CLA task vectors
    // On Type-1 CLAs the MVECT registers accept full 16-bit task addresses as
    // opposed to offsets used on older Type-0 CLAs
    //
    EALLOW;
    Cla1Regs.MVECT1 = (uint16_t)(&Cla1Task1);

    Cla1Regs.MCTL.bit.IACKE = 1;
    Cla1Regs.MIER.all = 0x00FF;

    PieVectTable.CLA1_1_INT  = &cla1Isr1;

    //
    PieCtrlRegs.PIEIER11.bit.INTx1 = 1;
    IER |= (M_INT11 );
    EDIS;
}

////
//// cla1Isr1 - CLA1 ISR 1
////
__interrupt void cla1Isr1 ()
{

    PieCtrlRegs.PIEACK.all = M_INT11;


}
