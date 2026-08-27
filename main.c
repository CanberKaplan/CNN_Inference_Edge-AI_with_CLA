

/**
 * main.c
 */
#include "F28x_Project.h"     // Includes everything: PieCtrlRegs, Cla1Regs, etc.
#include "cla_dense_layer_shared.h"
#include "weights.h"
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "convolution.h"
#include "definitions.h"
#include "feature_tables.h"
#include "golden_vector.h"
#include "adc.h"

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

#pragma DATA_SECTION(fResult,"CLADataLS0");
float fResult[10];  //output

#pragma DATA_SECTION(input_vector,"CLADataLS0");
int16_t input_vector[DENSE_LAYER_INPUT];

//#define MAX_ELEMENTS 11492 // 16 * 26 * 26
#pragma DATA_ALIGN(workspace, 8)
//float workspace[MAX_ELEMENTS] = {0.0f};
int16_t workspace[MAX_ELEMENTS] = {0.0f};

#pragma DATA_SECTION(weight,"CLADataLS1");
int16_t weight [DENSE_LAYER_WEIGHTS] = {    -10557, -15342, 10405, 6808, -12993, 6005, -12684, 12375, -12618, 15223,
                                            -4488, -8675, 13844, 1460, -15700, -8757, -16701, 5447, -9398, 9366,
                                            5605, -3292, 4044, -1855, -13106, -2075, -9393, 12489, 16667, -1982,
                                            -364, 17610, 17686, -5295, -11311, 10255, 1421, -39, -14915, 15632,
                                            -1381, -5409, 16147, 1014, -15563, 14551, 13234, 16475, -3457, 25742,
                                            6539, -32767, -4572, 1146, 6997, -30713, -29060, 11948, 19146, -28548,
                                            -2188, -132, 24807, -21608, -7714, -2474, 21050, -21375, -9797, 7391,
                                            15389, -15540, -13220, -1604, -3763, -8636, 19045, 12061, 10841, -5829,
                                            -7242, 5138, -4676, -15479, 4455, 6774, 2536, -4529, -19765, 6280,
                                            14643, -24995, 1627, 6334, 16424, -26296, -17651, 11683, -26385, 10384,
                                            -3081, 24499, -17348, -25640, -4424, -101, -25812, -12338, -16619, 29078,
                                            13259, -29460, -23868, 6542, -15057, -9469, -10329, 15490, -18227, -17664,
                                            -1850, 21939, -12505, 14511, 12145, 15630, 12859, 1366, -1407, -1193,
                                            -20945, -11426, 7627, 21046, 12984, -4868, -9377, 17018, -8178, 6860,
                                            -861, 24504, 11818, -13556, 4365, -3371, -25324, -9119, 17291, 12325,
                                            6952, -17757, 1204, 8351, -26863, -8791, -6234, 5252, 2619, -27211,
                                            7308, 18156, -7145, -1549, -757, -5582, 3317, -5465, -3118, -4680,
                                            -9891, -21963, -4957, -5974, -19911, -22430, 16113, 23037, -26455, -23094,
                                            -2672, 3776, 5119, -21315, 6538, 27560, -6978, -10695, 5612, 300,
                                            -21332, 3689, -22841, 9209, -9928, 20062, 16829, -9090, -24383, -9228,
                                            -23677, 16804, -12878, -4671, -4939, 17768, 3273, 3593, -14349, -10722,
                                            -23300, 3669, 8215, -9883, -10037, -14241, -13117, 14151, 2613, 20987,
                                            8811, 10871, -28989, -17222, 7878, -9001, -25565, 4636, 26978, 3949,
                                            -31138, -10731, 3959, 1672, -29936, 12817, -1885, -11735, -17966, 5758,
                                            10018, -24684, 14559, 21178, 12231, 18529, 2585, -3097, 7521, -18828,
                                            -2778, 27187, 6718, -7709, -12149, 3889, 23098, -25598, 7372, 21140,
                                            11469, 2471, -1767, 24499, 5639, -25419, -11540, 28419, -406, 10668,
                                            -1625, 14539, 15945, -21284, -6866, 4668, 13066, 17502, 4227, 12447,
                                            13546, -28645, 22166, 25677, -12427, -4217, -10667, 15335, -5611, 15136,
                                            7070, 5334, -6645, 13813, -8751, 1789, 16446, 5494, -7974, -12909,
                                            -12481, -2394, -6053, 13872, 15334, -2752, -13745, -10005, 9009, -203,
                                            -15743, 9986, 2893, 16955, 10704, 10042, 2121, -5302, -12699, -15762,
                                            13555, 15341, -12489, 10777, -11044, 3048, -13059, 8143, 10853, 14013,
                                            15324, 13575, 11833, 4030, -15197, -9637, 10618, -14931, -20546, 17321,
                                            9712, -9164, -31294, -58, -2704, -23300, -24466, 19522, -5551, -10505,
                                            -18300, 14364, 30517, -23706, -21030, -759, -13514, -10797, -12386, 12942,
                                            23481, 950, -16101, -2200, 14483, 15063, -6293, -583, 13219, -14946,
                                            -11127, 1402, -12949, 14588, -12532, 7479, 20342, -10654, -19916, 10575,
                                            -11172, 4054, -1618, 17677, -23177, -18231, -18425, 4550, -10463, -14928,
                                            7386, 6079, -15554, -28515, -22052, 4128, 1370, -19276, 4752, 12292,
                                            -22688, -22867, -22704, -6476, -6201, -6507, 9774, -5535, -51, -19532,
                                            -2036, -3772, 24095, -24684, 11517, 2228, -16371, 1541, -2405, 697,
                                            5740, -26491, 7021, 2420, -12693, -617, -15886, 1153, 13375, -5544,
                                            -8263, -2046, 6025, -21635, 13562, -5793, -637, 8530, 6928, 24113,
                                            -8038, -3290, 20460, 26712, 6789, 3376, -825, 16047, 7422, -205,
                                            25547, -1377, -12848, -23952, 21219, -5085, 4091, -25896, 7800, 23094,
                                            13540, 1596, 23240, 238, -7604, 2051, 17846, 10823, -9865, -5300,
                                            19721, 29443, 19445, 7600, -3345, 370, -7228, -12131, -5922, 16500,
                                            1724, -15992, 17327, 6394, 6405, 5735, -11776, -13326, -20874, -17672,
                                            7086, 1818, 6128, -1703, -13512, 18327, 2447, -16782, 11874, 6692,
                                            3147, -11893, -16624, 10266, -29120, -3551, 1272, -2642, -14643, -16498,
                                            -10902, -13068, -11688, -27279, 18705, -3049, -1351, -2531, 1667, 14268,
                                            -10029, -7157, 16376, -16983, -21415, -28389, -13538, 9265, -7851, 19454,
                                            1565, -4092, -16960, -10695, -3171, -27067, -17255, 18986, -816, 632,
                                            -1497, -5802, 2548, -5873, 6712, 20412, -21483, -16808, -18962, 7000,
                                            8560, -24108, -23952, -172, -15463, -15665, -6611, -7116, -6369, 774,
                                            2671, 27877, -10428, -6211, -11481, 4225, 7542, -21296, -17747, -659,
                                            -17804, -11764, -10753, 16685, 2184, -9315, 13151, -15989, 9092, -2237,
                                            10714, -15936, 6640, -10986, -2069, 13692, 10535, 2710, 3423, 3959,
                                            -13172, -4923, 3871, 14231, 15084, 9363, 2102, -15858, -15288, 2006,
                                            17627, -2493, -4257, 13576, -8712, -5503, 4617, 12702, 10378, 8185,
                                            -15741, 11835, -13365, -16152, 9347, 15763, 16152, 802, 14764, -8441,
                                            -17708, -2431, -8384, -6271, -13290, -1481, 579, -13432, -1903, 6339,
                                            -3446, -15182, -13009, -25660, -15429, -4941, -24440, -22480, -19334, -17402,
                                            -23470, 1845, 995, 13664, -3959, -15866, 2375, -15458, -10611, -10135,
                                            5018, 5048, -19312, -6135, 9151, -2610, -25864, -10449, -16205, -6425,
                                            -8824, -1253, 8621, -9323, -7719, -25916, -20305, 7399, -842, -20775,
                                            365, -10486, 4453, 23790, 19241, -2042, 2004, -6771, 19128, 20040,
                                            28204, 22491, 7954, 13840, -16907, 13487, -3810, 3247, -1776, 20900,
                                            30976, 24391, -15951, 18785, 18734, 17894, 10303, 15023, 29080, 20290,
                                            -4654, -5751, 20273, 19753, 7613, 26582, 18497, 12012, -7792, -335,
                                            15776, 16425, 2408, -1872, 10167, -301, 7178, 7701, -5490, 10977,
                                            -15759, -475, -9683, -4507, -13223, 10029, -14153, -13477, -9461, -6775,
                                            -5633, -12791, -17662, 1123, -946, 2990, -3933, 5346, -11110, 2795,
                                            -7736, -6611, 12795, -28198, -20175, -22168, 8384, -19643, -18189, -3193,
                                            -13444, 2290, 2582, -16781, -1281, -20291, -13188, -780, -8190, 3061,
                                            -2479, 5651, -4705, 5694, 1598, -16847, -16442, -25069, 15337, 23676,
                                            5398, -9771, -18401, -2416, 18458, 26660, 2448, 17764, 13610, 6848,
                                            -7337, 21099, 5236, 24372, 8548, -2837, 21877, -3930, -20206, 15455,
                                            18536, -7524, 28654, 24380, 13634, -6800, -27968, 22312, -6194, -7166,
                                            26576, 9098, -1675, 10948, -9310, -2126, 18182, 26475, 13510, 27375,
                                            845, -20086, -5431, 12528, 16420, 12952, -10670, 1561, -4613, -9675,
                                            -1831, 5560, -16006, 5552, -12948, 8591, 842, -4899, -1129, -21422,
                                            6767, -2276, -9212, 8387, -15633, -16389, 10310, -15707, -18731, -7960,
                                            12775, 18629, -9645, 1176, -2236, -11391, -175, 4924, 1913, -9116,
                                            -1049, -12278, 17445, -5573, 2296, -8387, 6462, 16271, 10762, 6859,
                                            15390, -17232, 10469, 2882
                                        };

#pragma DATA_SECTION(adcData0, "ramgs0");
//#pragma DATA_SECTION(adcData1, "ramgs0");
Uint16 adcData0[RESULTS_BUFFER_SIZE];
//Uint16 adcData1[RESULTS_BUFFER_SIZE];
volatile Uint16 done;
#endif //__cplusplus

void CLA_runTest(void);
void CLA_configClaMemory(void);
void CLA_initCpu1Cla1(void);

static float features[FEAT_N_FRAMES][FEAT_N_OUT];   /* 32x12 */
__interrupt void cla1Isr1();

void mfcc_extract(const unsigned int* adc_buf,
                        float out[FEAT_N_FRAMES][FEAT_N_OUT]);

int main(void)
{
    Uint16 resultsIndex;

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

    // ISR for ADCA INT1 - occurs after first conversion
    // ISR for DMA ch1 - occurs when DMA transfer is complete
    //
        EALLOW;
        PieVectTable.ADCA1_INT = &adca1_isr;
        PieVectTable.DMA_CH1_INT = &dmach1_isr;
        EDIS;

    //
    // Enable specific CPU interrupts: INT1 for ADCs and INT7 for DMA
    //
        IER |= M_INT1;
        IER |= M_INT7;

    //
    // Enable specific PIE interrupts
    //
    // ADCA INT1 - Group 1, interrupt 1
    // DMA interrupt - Group 7, interrupt 1
    //
        PieCtrlRegs.PIEIER1.bit.INTx1 = 1;
        PieCtrlRegs.PIEIER7.bit.INTx1 = 1;


        //
        // Stop the ePWM clock
        //
            EALLOW;
            CpuSysRegs.PCLKCR0.bit.TBCLKSYNC = 0;
            EDIS;

        //
        // Call the set up function for ePWM 2
        //
            ConfigureEPWM();

        //
        // Start the ePWM clock
        //
            EALLOW;
            CpuSysRegs.PCLKCR0.bit.TBCLKSYNC = 1;
            EDIS;

        //
        // Configure the ADC and power it up
        //
            ConfigureADC();

        //
        // Setup the ADC for continuous conversions on channels A3 and B3
        //
            SetupADCContinuous(&AdcaRegs, 3);
            SetupADCContinuous(&AdcbRegs, 3);

        //
        // Initialize the DMA
        //
            DMAInit();



    CLA_configClaMemory();
    CLA_initCpu1Cla1();

    EINT;  // Enable Global interrupt INTM
    ERTM;  // Enable Global realtime interrupt DBGM
    //
    // Initialize results buffer
    //
        for(resultsIndex = 0; resultsIndex < RESULTS_BUFFER_SIZE; resultsIndex++)
        {
            adcData0[resultsIndex] = 0;

        }

    //
    // Clearing all pending interrupt flags
    //
        EALLOW;

        DmaRegs.CH1.CONTROL.bit.PERINTCLR = 1;
        DmaRegs.CH2.CONTROL.bit.PERINTCLR = 1;
        AdcaRegs.ADCINTFLGCLR.all = 0x3;
        AdcbRegs.ADCINTFLGCLR.all = 0x3;
        EPwm2Regs.ETCNTINITCTL.bit.SOCAINITFRC = 1;
        EPwm2Regs.ETCLR.bit.SOCA = 1;

    //
    // Enable continuous operation by setting the last SOC to re-trigger the first
    //
        AdcaRegs.ADCINTSOCSEL1.bit.SOC0 = 2;
        AdcbRegs.ADCINTSOCSEL1.bit.SOC0 = 2;

        EDIS;

    //
    // Start DMA
    //
        done = 0;
        StartDMACH1();
        StartDMACH2();

    //
    // Finally, enable the SOCA trigger from ePWM. This will kick off
    // conversions at the next ePWM event.
    //
        EPwm2Regs.ETSEL.bit.SOCAEN = 1;

    for(;;)
     {
        GPIO_WritePin(18, 0);
        static uint32_t i,f,o;
        mfcc_extract((const unsigned int*)golden_adc, features);

        int idx = 0;
                for ( f = 0; f < FEAT_N_FRAMES; f++) {
                    for ( o = 0; o < FEAT_N_OUT; o++) {
                        float scaled = features[f][o] * INPUT_SCALE;
                        if (scaled > 32767.0f) scaled = 32767.0f;
                        if (scaled < -32768.0f) scaled = -32768.0f;
                        input_image[idx++] = (int16_t)scaled;
                    }
                }
        GPIO_WritePin(18, 1);

        GPIO_WritePin(19, 0);

        convolution_int16(input_image, workspace, conv1_weights, conv1_bias, IMAGE_H,  IMAGE_W, CONV1_WEIGHT_SCALE);

        relu_activation_int16( workspace,  MAX_ELEMENTS);

        max_pooling_2x1_int16(workspace, &workspace[CONV1_SIZE],
                                      CONV1_OUT_H, CONV1_OUT_W, NUM_FILTERS);

        convolution_2_int16(
                            &workspace[CONV1_SIZE],               // Girdi: Pool1 çıktısının bulunduğu yer
                            &workspace[CONV1_SIZE + POOL1_SIZE],  // Çıktı: Conv2 sonucunun yazılacağı yeni boş alan
                            conv2_weights,
                            conv2_bias,
                            POOL1_OUT_H,                      // Girdi Yüksekliği
                            POOL1_OUT_W,                      // Girdi Genişliği
                            CONV2_WEIGHT_SCALE
                        );

        relu_activation_int16(&workspace[CONV1_SIZE + POOL1_SIZE], CONV2_SIZE);

        max_pooling_2x1_int16(&workspace[CONV1_SIZE + POOL1_SIZE], input_vector,
                                      CONV2_OUT_H, CONV2_OUT_W, NUM_FILTERS);
        GPIO_WritePin(19, 1);

        CLA_runTest();

//        if(done == 1)
//                {
//                    static uint32_t f,o;
//
//                    // DİKKAT: golden_adc YERİNE adcData0 KULLANILIYOR
//                    mfcc_extract((const unsigned int*)adcData0, features);
//
//                    int idx = 0;
//                    for ( f = 0; f < FEAT_N_FRAMES; f++) {
//                        for ( o = 0; o < FEAT_N_OUT; o++) {
//                            float scaled = features[f][o] * INPUT_SCALE;
//                            if (scaled > 32767.0f) scaled = 32767.0f;
//                            if (scaled < -32768.0f) scaled = -32768.0f;
//                            input_image[idx++] = (int16_t)scaled;
//                        }
//                    }
//
//                    convolution_int16(input_image, workspace, conv1_weights, conv1_bias, IMAGE_H,  IMAGE_W, CONV1_WEIGHT_SCALE);
//                    relu_activation_int16( workspace,  MAX_ELEMENTS);
//                    max_pooling_2x1_int16(workspace, &workspace[CONV1_SIZE], CONV1_OUT_H, CONV1_OUT_W, NUM_FILTERS);
//
//                    convolution_2_int16(&workspace[CONV1_SIZE], &workspace[CONV1_SIZE + POOL1_SIZE], conv2_weights, conv2_bias, POOL1_OUT_H, POOL1_OUT_W, CONV2_WEIGHT_SCALE);
//                    relu_activation_int16(&workspace[CONV1_SIZE + POOL1_SIZE], CONV2_SIZE);
//                    max_pooling_2x1_int16(&workspace[CONV1_SIZE + POOL1_SIZE], input_vector, CONV2_OUT_H, CONV2_OUT_W, NUM_FILTERS);
//
//                    // CLA işlemini tetikle
//                    CLA_runTest();
//
//                    // 3. EKSİK: BAYRAĞI SIFIRLA
//                    // İşlemler bitti. DMA'nın bir sonraki 1024'ü doldurmasını beklemek için bayrağı indir.
//                    done = 0;
//                }

     }

}



void CLA_runTest(void)
{


    Cla1ForceTask1();

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
//    MemCfgRegs.LSxMSEL.bit.MSEL_LS4 = 1;
//    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS4 = 1; // program memory
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

    MemCfgRegs.LSxMSEL.bit.MSEL_LS3 = 1;
    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS3 = 0;

    MemCfgRegs.LSxMSEL.bit.MSEL_LS4 = 1;
    MemCfgRegs.LSxCLAPGM.bit.CLAPGM_LS4 = 0;
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
