

/**
 * main.c
 */
#include "F28x_Project.h"     // Includes everything: PieCtrlRegs, Cla1Regs, etc.
#include "cla_dense_layer_shared.h"
// GUNCELLEME (asama bazli bolusum): hangi asamanin CLA'da hangisinin CPU'da
// calisacagini belirler; weights.h'teki DATA_SECTION pragma'larini da
// suruyor, bu yuzden ONCE include edilmeli.
#include "cla_pipeline_config.h"
#include "weights.h"
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "convolution.h"
#include "definitions.h"
#include "feature_tables.h"
// GUNCELLEME: gercek ADC/DMA yolu henuz hazir degil (ayri bir gorusme
// konusu -- arabellek boyutu eski SLICE_LEN'e gore, cift tamponlama yok).
// Bu asamada ucu sinifin (Healthy/Bearing/Propeller) golden vector'u AYNI
// ANDA dahil edilip runtime'da aralarinda gecis yapilarak pipeline'in
// (mfcc_extract -> quantize -> CLA conv1+conv2+FC) girdinin degismesini
// dogru yakalayip yakalamadigi test ediliyor -- gercek ADC olmadan.
#include "golden_vector_healthy.h"
#include "golden_vector_bearing.h"
#include "golden_vector_propeller.h"
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
float fResult[NUM_CLASSES];  //output

#pragma DATA_SECTION(input_vector,"CLADataLS0");
int16_t input_vector[DENSE_LAYER_INPUT];

//#define MAX_ELEMENTS 11492 // 16 * 26 * 26
// GUNCELLEME (asama bazli bolusum): workspace (pool1 + pool2 ciktilari) HER
// konfigurasyonda CLA veri RAM'inde tutulur -- hicbir asama CLA'da olmasa
// bile. Sebep olculdu (TI cl2000 22.6.1 ile 8 kombinasyonun hepsi derlendi):
//   * CPU RAM'i (RAMGS_TOTAL_CNN, 45056 word) MFCC tamponlariyla zaten
//     42688 word dolu (mfcc_extract::x[] tek basina 33792 word). workspace
//     (2532 word) oraya dusurulunce ".ebss will not fit" (#10099) ile link
//     hatasi alinir (cla_pipeline_config.h: 0/0/0).
//   * CLA veri RAM'i (RAMLS_0_1_2_3_4, 10240 word) hicbir kombinasyonda
//     sikisik degil (en fazla 6858 word, hepsi CLA'dayken).
// CPU, CLA veri RAM'ini (LSx, MSEL=CLA) zaten okuyup yazabiliyor; kisit tek
// yonlu -- CLA bu havuzun DISINA erisemiyor. Yani tamponu burada tutmak her
// kombinasyonda dogru, CPU'ya tasimanin kazanci yok, bedeli link hatasi.
// (Asamaya gore kosullu olan sadece AGIRLIKLAR: CPU'dayken flash'ta kalirlar
// ve RAM harcamazlar.)
#pragma DATA_SECTION(workspace, "CLADataLS0");
#pragma DATA_ALIGN(workspace, 8)
//float workspace[MAX_ELEMENTS] = {0.0f};
 int16_t workspace[MAX_ELEMENTS] = {0.0f};

// GUNCELLEME: 864 elemanli eski (32x12 girisli model) veri, 2016 elemanli
// yeni (65x12 girisli, dataset__eval_heldout_test) fc_weights ile degistirildi.
// GUNCELLEME 2 (GERI ALINDI): "CLADataLS1" pragma'si tekrar eklendi -- FC
// hesabi yeniden CLA'da (bkz. dense_layer.cla). Asil sorun MAX_ELEMENTS'in
// (workspace) eski, fused-olmayan tasarimdan kalma yanlis formulle ~9000
// byte israf etmesiydi (definitions.h::MAX_ELEMENTS notu); o duzeltilince
// weight[] (4032 byte) CLA'nin dar belleğine rahatlikla sigiyor.
// GUNCELLEME (asama bazli bolusum): FC agirliklari (4032 byte) yalnizca
// dense asamasi CLA'dayken CLA RAM'inde tutulur; CPU'ya alinirsa flash'ta
// kalir ve CLA havuzunda 4032 byte yer acilir.
#if CLA_STAGE_DENSE
#pragma DATA_SECTION(weight,"CLADataLS1");
#endif
const int16_t weight [DENSE_LAYER_WEIGHTS] = {
    -7664, -28530, -17363, -3650, -19408, 17616, -3049, 9839, -12674, -20266,
    -22285, -6135, -18696, 12146, -24552, -5964, -7703, -16406, -17300, -3210,
    -8078, 11920, -24587, -6071, -18547, -24055, -9729, -8252, -11466, 5878,
    -6244, 6725, -5941, -27118, -1398, 20299, -5311, 12063, -19325, 16718,
    -18352, -22135, -22919, -598, -4126, 17470, -19234, 9102, -8725, -28432,
    -4689, 5905, 9351, 1327, -9234, 8932, -17074, -10052, -17367, 10788,
    13937, 3530, -7958, 5058, -29297, -12148, -7571, 1212, 3670, 3942,
    -10528, 6534, -8648, -23654, -9968, 6684, 2461, 9191, -7779, 18009,
    -8883, -13809, -240, 16353, -5289, 14519, -2456, 9895, -23904, -29748,
    -16317, 14338, 2031, 17616, -5892, 9318, -4590, -20505, -8663, 15657,
    -11476, 12545, -10239, -5622, -20289, -14875, -8451, 7246, -8478, 21890,
    -13825, -6188, -20232, 7936, 13886, -11220, -1530, -906, 19385, -22393,
    -9178, -1345, 18345, -2841, 5389, -12488, 19219, -25928, -6337, -2449,
    25666, -16383, 9604, -21966, 5811, -6946, -25560, -8878, 10185, -16792,
    332, -8084, 26229, -1710, -14088, 958, 28944, -2907, -7827, 6733,
    10989, -7679, -16038, -5144, 31249, -15321, 461, -6310, 3968, -3734,
    -13590, 11604, 19191, -2967, -20149, 756, 17213, -23300, -1633, 4926,
    28505, -3333, 10704, -18551, 2814, -18736, -17833, -8687, 18951, -21539,
    -13999, 4032, 8474, -14406, -7744, -3581, 10394, -18676, -4726, -7163,
    29255, -7359, -18831, -6490, 17473, -12498, -13892, -15026, 15371, -16702,
    -6398, 17264, 9117, -24709, 4177, -508, 6221, -8457, -24285, 14622,
    10341, 3231, -6512, -11322, 27302, -12139, -26885, -1307, 7076, -9775,
    -9982, 1327, 23099, -12845, 16027, -12149, -1029, 11262, -14727, 21412,
    -1757, 16148, 12169, 1960, 22911, 10900, -15241, 20645, -10853, 19184,
    1327, 1496, 8318, 12188, 5606, 13951, -23483, 28352, 22087, 3317,
    9906, 23445, -9480, -401, -17410, 16762, 6099, -21482, -1811, 14193,
    2368, 2271, -11728, 14810, 20428, -14488, 4822, 1751, -183, 6627,
    -20036, 19244, 15138, -9505, 1311, 310, -13339, 18282, 2927, 24449,
    11103, -3439, 12182, 7368, -11790, 14582, -9778, 21442, 3172, -3606,
    7381, 6869, 306, 5884, -11768, 19797, 11944, -18498, 5053, 16103,
    -11877, 18024, -8045, 26720, -860, -21522, 10873, 9798, -12389, 2667,
    -18103, 29136, 23811, -23597, -3509, 12792, -977, 765, -16711, 18860,
    71, -14723, 6733, 25199, -1393, 22810, 4508, 17277, 25467, -1478,
    11674, 7901, -10729, 736, -652, 22905, 9681, 18710, 15348, -3956,
    642, 3875, 12347, 4597, 9182, 15853, 8670, -22119, -11159, 883,
    18536, -2584, 10111, 11986, 9413, -12237, -14001, 2020, 220, -2238,
    6351, 3484, 15069, -1855, 1091, 6905, 6003, -9746, 6972, 17127,
    5790, -20542, 1926, -6027, 2524, -13346, -11715, 8098, 19516, -5056,
    -13183, -7475, 8362, -567, 8309, 6991, 12094, 3686, -1552, 3311,
    12466, -3789, -13714, 25391, 3319, -857, 329, 12949, 17997, -9518,
    -9985, 13840, 2852, 1812, 393, -3379, 579, 8769, 12332, 24874,
    7661, -20412, -8832, 3845, 10919, -14902, 4001, 13529, 11864, -1661,
    11621, 15662, 11584, 8090, -11687, 11916, 27619, -18371, -11057, 3604,
    26471, 1922, -14358, 15064, 25058, -11134, -13674, 7782, 4814, -8536,
    -6102, 27019, 16447, -16827, 1332, 15461, 29446, 17124, -8760, -23339,
    -14389, -18152, -9118, -2090, 15400, -20907, -20371, -20882, -2815, -23455,
    -4086, -6937, 16214, -13918, -12875, -3556, -14613, -23947, -14237, -1160,
    9452, -7559, -11325, -11323, -6811, -20194, -12507, -19901, 14914, -2618,
    -8328, -14625, -10377, -25319, 6663, -4300, -779, -19000, -19249, -9567,
    -19587, -3648, 6654, -9529, 605, -4873, -23033, -10676, -2345, -10918,
    -9949, -25087, -3753, -17888, -20826, -2344, -2117, -17201, -13649, -22808,
    13329, -15051, -27822, -5908, -11352, -1491, 4735, -6255, 7140, -18878,
    -25767, -24186, -3727, -3183, -12080, -4099, -2006, -14133, -25931, -6057,
    -5454, -4087, 8725, -4447, 17413, -12149, -27784, -19990, -16402, -20220,
    -7261, -10342, 12038, -27057, -28096, -24254, -26453, -27521, -15121, -7243,
    -983, -26594, -18309, -11914, -1614, -27211, -8508, -24450, -6599, -28853,
    10833, -9935, -9172, 5991, -15389, -4495, -8445, -200, 16559, -22075,
    -17074, -8289, -11898, -2142, -27860, -893, 12065, -13508, -11703, -2729,
    -19326, 5933, -27311, -3800, 18964, -7837, -23592, -4666, -21140, 16948,
    -29325, -3392, 15481, -11090, -19161, -12254, -23542, 16296, -20197, -6655,
    14708, -5066, -23906, -8126, -24624, 8839, -25993, 2135, 15864, -19071,
    -5775, -8466, -9740, 13858, -4910, -10986, 17776, -13946, -18102, 8759,
    -9109, 15528, -17802, -14540, 15392, -12081, -16427, 8841, -17808, 14334,
    -11636, 3088, 2215, -14815, -18103, -13705, -9155, -7604, -20375, 11144,
    9183, -932, -28599, 2950, -4326, 18331, -13534, 5549, 3938, -8830,
    -18965, 3006, -421, -967, -29398, -11464, 21469, -24050, -20318, -14812,
    -13222, 15843, -9929, 3982, 20872, -19321, -28115, 4890, -5929, 10513,
    -26868, -2304, 1917, 23261, -2501, 1675, -3113, -22079, 3980, -14211,
    -16909, 14192, 1596, -10511, 13728, -3922, 23086, -923, -16728, -905,
    -16783, -12834, 5050, -7288, 20670, -5363, -17090, 22691, 1365, -12547,
    11872, -2945, 19190, -12976, -20512, 16465, -20429, -2915, 4298, -10329,
    6091, -18330, -14736, 184, -11193, -22191, 14094, -12158, 16413, -5685,
    -3157, 20410, -8313, -5765, 15307, -22325, 1270, -25781, -16170, 16728,
    -9040, -4104, 14270, -13382, 4176, -24052, -17050, -133, -16241, -4060,
    13205, -9872, 15504, -21829, -10876, 14793, 323, -3010, -4153, -23400,
    -395, -12390, -7867, 1171, -21842, -19044, -570, -21618, 11077, -22938,
    -18836, 19267, -1702, 679, 4931, -16551, 7544, -13063, -7204, 3411,
    -14954, -21040, 725, -19731, 8906, -4945, -12474, 11895, -2845, -21582,
    9643, -16574, 20962, -24908, 13840, -7580, -12278, 10974, 2022, 5902,
    -7743, 7575, -34, -2631, -7033, 12273, 5974, 10670, -8770, -7007,
    -1736, 1853, 2321, 81, -8626, 13329, -15334, 1531, 12409, 10100,
    -2950, 5825, -17350, 10684, -19793, -4855, 15674, 11919, 1558, 1184,
    6030, -2130, -21579, 4998, 2708, -2005, -8688, 10912, 2400, -7189,
    -4160, 17305, 11174, -2768, -8747, -6817, -9105, -2438, -23092, -5950,
    220, -8770, -16975, 13650, 4342, 12551, -7231, 725, 16747, -7685,
    -7079, 5344, 8015, 4110, -17629, 7240, 19079, 4620, -18050, 5737,
    -7673, -2384, -11198, 10816, 20198, -969, -5047, -7726, -13889, 7668,
    -3892, -6623, 15053, -2222, -17411, 10931, -1097, -4965, -5088, -3525,
    22063, 10201, -470, 8469, -9027, 8917, -17436, 14181, 4698, -1697,
    -21228, 8095, 7010, 15416, -15796, 16748, 31809, 28036, 15789, 11274,
    16931, 13603, 8481, 4920, 32767, 28817, 11794, 21839, 8815, 11279,
    7308, 8638, 29934, 27912, 12311, -2200, -3606, -4498, 17286, 20015,
    11936, 19541, 13541, 21698, -6736, 8112, 2848, 24296, 10031, 14530,
    18076, 9163, -3920, 4074, -1553, 16322, 21421, 26946, 23618, 19415,
    -12550, 13031, 7408, 13337, 30065, 3912, 10147, 1565, 3255, -7429,
    -4165, 17158, 13484, 29754, 25398, 4513, -3824, -2773, 1818, 19723,
    19011, 28950, 23367, 20870, -783, 17563, 17664, 9604, 10698, 7319,
    16484, 18301, 30, -1081, 886, 4322, 24893, 32502, 6971, 10751,
    3217, 13243, 8878, 25067, 15298, 29388, 13145, -1171, 9374, 3457,
    -3691, 18951, 24868, 6037, 26551, 16750, 3507, 587, -6719, 21931,
    5739, 5059, 25320, 5287, -8734, 14696, 6599, 4931, 7822, 21939,
    -1547, 19257, 4908, -9500, 4376, 16212, 4216, 8520, -9860, 15211,
    5662, 581, -4590, 25255, -9281, 14432, -15753, 15490, 10911, 3497,
    -3198, 3824, 9792, 6948, -16980, 11110, 3598, -3948, 4511, 3371,
    -10195, 7914, -4197, 20404, 2260, 6151, -7953, 6285, 4387, 18235,
    -11057, 1116, 11299, -2379, 7366, 11680, 79, 5633, -7277, 20601,
    22941, 9107, 6356, 5044, 10800, 12136, 3449, 4254, 19479, -5374,
    1325, 16449, -396, -24, 7774, 13596, -1895, -1648, 6694, 4363,
    -7669, 13367, -474, 21090, 2509, -11523, 9479, 24960, 6593, 5150,
    -3079, -4835, 1237, -1270, 11633, 22321, 2358, 12197, -7074, 8424,
    10965, -1218, -1050, -1642, 4200, 14290, -7132, 2978, 8339, -5823,
    -7677, 15453, 8522, 12306, -9147, 10026, 20196, -3531, 2010, 4367,
    -10028, -7888, 3662, -11648, -9012, 2815, -17596, -23192, -15584, -17458,
    -8562, -4260, 6779, 9210, -5330, -10102, -5999, 2875, -5331, -17813,
    -6784, -18218, -20461, -26152, -3904, -14477, 11399, -18843, -4170, -7362,
    -25983, -11992, -2735, -422, -9622, -23565, -6159, -7667, -20613, -3730,
    -16397, -5869, 7824, -2944, -14, -9163, -16640, -16827, -13339, -1154,
    8915, -8237, 2544, 5566, -5663, -16332, -12999, 4436, 3241, -2618,
    -8382, -12567, -25347, -24897, -15967, -8135, -4246, 1851, 6869, -7530,
    -20277, -2174, 4532, -14585, 3908, -5509, -927, -3595, -19041, -28828,
    -11179, 975, -3503, -24247, -6148, -15372, -8972, -20584, -1303, -19948,
    12764, 1917, -12925, 3851, -5628, -9564, -8029, -17450, 486, -22524,
    -4630, 159, -3716, -9183, -14576, -4242, 17751, -7524, 1088, -13407,
    -25171, -12588, -22149, -10432, -12345, -3736, -492, -8619, 2304, -11267,
    -3997, -23712, -12836, 4660, 15381, -12251, -4176, 1527, -4898, -16947,
    -11101, 12285, 12132, 1449, -11845, 3387, -13699, -9566, -14534, 12387,
    1771, -13766, -19493, 1661, -21640, -14286, -433, 443, -1480, -5716,
    -3377, -9245, -9961, -14935, -6827, 9340, 14612, -17645, -16447, -8263,
    -12308, -21809, -6344, 16697, 8503, -1388, -7549, -6945, -14405, -12091,
    4752, -1085, -4820, -12597, -7593, 3109, -23366, -25067, -7798, 2873,
    -2663, -9405, -151, -13892, -21221, -13904, 3922, 12392, -1106, -585,
    -12398, 1756, -11313, -6544, 5179, 8517, 6557, -10146, -17193, 2927,
    -5722, -17741, -745, 6448, 11067, 2100, -11042, -14242, -24173, -27048,
    4389, 16151, -1722, -22457, -7095, -13437, -12469, -20054, 768, 9058,
    4779, -379, -11783, -9324
};

#pragma DATA_SECTION(adcData0, "ramgs0");
//#pragma DATA_SECTION(adcData1, "ramgs0");
Uint16 adcData0[RESULTS_BUFFER_SIZE];
//Uint16 adcData1[RESULTS_BUFFER_SIZE];
volatile Uint16 done;
// EKLENDI: CLA1_1_INT (Cla1Task1 BITTIGINDE ates lenen kesme) bunu 1 yapar --
// main loop artik fResult[]'i CLA henuz hesaplamayi bitirmeden okumuyor.
// Eskiden boyle bir senkronizasyon yoktu.
volatile uint16_t cla_done = 0;
#endif //__cplusplus

void CLA_runTest(void);
void CLA_configClaMemory(void);
void CLA_initCpu1Cla1(void);

static float features[FEAT_N_FRAMES][FEAT_N_OUT];   /* 65x12 (eskiden 32x12 -- SLICE_LEN 2x oldu) */
__interrupt void cla1Isr1();

// EKLENDI: golden-vector dongusu -- gercek ADC yerine, uc sinifin
// (Healthy/Bearing/Propeller) onceden hesaplanmis golden_adc dizisi
// sirayla beslenip pipeline'in giris degisimini dogru yakalayip
// yakalamadigi (CLA sonucu -> beklenen sinifla eslesiyor mu) test edilir.
static const uint16_t *const golden_inputs[NUM_CLASSES] = {
    golden_adc_healthy, golden_adc_bearing, golden_adc_propeller
};
static const char *const golden_names[NUM_CLASSES] = { "Healthy", "Bearing", "Propeller" };
// CCS'in Expressions/Watch penceresinden debug sirasinda canli izlemek
// icin: hangi giris beslendi, model ne tahmin etti, dogru mu.
volatile uint16_t g_expected_class = 0;
// KALDIRILDI (paralellik modu): g_predicted_class, g_correct_count,
// g_total_count ve g_last_logits[][]. Hepsi fResult'i okuyordu; ana dongu
// artik CLA'yi beklemedigi icin o okuma yaris olurdu. Tanimlarini birakmak
// daha kotuydu -- hic yazilmadiklari icin Watch penceresinde sonsuza kadar
// 0 durup modelin hata verdigini dusundururlerdi. Dogrulama isteyen,
// bekleyen surumu kullanmali (git log: "golden-vector dongusuyle
// giris-degisimi dogrulamasi"). tools/board_golden_test de o surume bagli.

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

    // EKLENDI: sirayla beslenecek sinif indeksi (0=Healthy,1=Bearing,2=Propeller,
    // sonra basa sarar). static: fonksiyon disina tasinsa da her cagrida degerini korur.
    static uint16_t golden_cycle_idx = 0;

    for(;;)
     {
        GPIO_WritePin(18, 0);
        static uint32_t i,f,o,k;
        // GUNCELLEME: sabit golden_adc yerine, dongudeki mevcut sinifin
        // girisi besleniyor -- pipeline'in giris DEGISTIKCE dogru sinifa
        // gecis yapip yapmadigini (gercek ADC olmadan) test etmek icin.
        g_expected_class = golden_cycle_idx;
        mfcc_extract((const unsigned int*)golden_inputs[golden_cycle_idx], features);

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

        // GPIO19: artik SURE degil, DEVIR TESLIM ANI. Asagida CLA'nin
        // bitmesini beklemedigimiz icin bu iki kenar arasi yalnizca tetikleme
        // komutlari kadar (mikrosaniyeler). Inference suresini olcmek icin
        // GPIO67'ye bakin -- onu Cla1Task1 kendi icinden suruyor.
        GPIO_WritePin(19, 0);

        // PARALELLIK MODU: CLA tetiklenir ve BEKLENMEZ. CPU dongunun basina
        // donup bir sonraki pencerenin MFCC'sini hesaplamaya baslar; CLA bu
        // sirada onceki pencerenin agini isletir. Iki pin ayni anda dusuk
        // gorunur -- olculmek istenen ortusme budur:
        //     GPIO18 dusuk = C28x oznitelik cikariyor
        //     GPIO67 dusuk = CLA inference yapiyor   (Cla1Task1 suruyor)
        //
        // BUNUN BEDELI: sonucu kimse beklemediginden fResult'i okumak yaris
        // olur -- argmax ve dogruluk sayaclari bu yuzden kaldirildi. Bu
        // yapilandirma ZAMANLAMA gozlemi icindir; dogrulama isteyen, bekleyen
        // surumu kullanmali (git log: "golden-vector dongusuyle
        // giris-degisimi dogrulamasi").
        //
        // TEK TAMPON, VE NEDEN YETIYOR: CLA ile CPU'nun paylastigi tek
        // degistirilebilir tampon input_image[]. workspace[] ve fResult[]
        // yalnizca CLA'nin, agirliklar sabit, ve CPU artik fResult'i
        // okumuyor.
        //
        // input_image[] pencerenin SONUNDA yaziliyor (mfcc_extract bittikten
        // sonraki int16 donusum dongusu). CLA hemen ardindan tetiklendigi
        // icin, bir sonraki yazmaya kadar elinde TAM BIR oznitelik cikarma
        // periyodu var. Kosul bu kadar basit:
        //
        //     inference <= oznitelik cikarimi
        //      34 ms    <= 202 ms        -> 168 ms pay  (4 kanal, olculen)
        //
        // Yani eskiden "verimlilik" sanilan esik, tek tamponlu bu tasarimin
        // DOGRULUK siniri. Asilirsa kod cokmez, hata da vermez: CLA hala
        // onceki pencereyi okurken CPU ustune yeni pencereyi yazar ve sonuc
        // sessizce bozulur. O noktada input_image ping-pong tamponlanmali.
        cla_done = 0;
        CLA_runTest();
        GPIO_WritePin(19, 1);

        golden_cycle_idx++;
        if (golden_cycle_idx >= NUM_CLASSES) golden_cycle_idx = 0;

//        CLA_runTest();

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



// GUNCELLEME (asama bazli bolusum): inference artik tek bir sabit cagri
// degil -- her asama (S1: conv1+pool1, S2: conv2+pool2, S3: dense)
// cla_pipeline_config.h'a gore ya CPU fonksiyonuyla ya da CLA gorevi
// tetiklenerek yurutuluyor. MFCC bu fonksiyonun DISINDA, her zaman CPU'da
// (main dongusundeki mfcc_extract cagrisi).
//
// Iki mod var:
//  * CLA_SINGLE_TASK (uc asama da CLA'da): eskisi gibi tek tetik
//    (Cla1ForceTask1, ASENKRON) -- CPU beklemez, isin bittigini cla_done
//    ISR'i bildirir. Cagiran taraf sonucu kullanmadan once cla_done'i
//    beklemeli (main dongusunde oyle yapiliyor).
//  * Karma dagilim: asamalar sirayla, veri bagimliligi nedeniyle
//    BLOKLAYICI olarak (Cla1ForceTaskNandWait) calistirilir; bitince
//    cla_done=1 yapilir ki cagiran tarafin bekleme mantigi ayni kalsin.
void CLA_runTest(void)
{
#if CLA_SINGLE_TASK
    Cla1ForceTask1();          // tum inference tek gorevde, asenkron
#else
  #if CLA_STAGE_CONV1_POOL1
    Cla1ForceTask1andWait();
  #else
    conv1_pool_fused_cpu(input_image, workspace, conv1_weights, conv1_bias);
  #endif

  #if CLA_STAGE_CONV2_POOL2
    Cla1ForceTask2andWait();
  #else
    conv2_pool_fused_cpu(workspace, &workspace[POOL1_SIZE], conv2_weights, conv2_bias);
  #endif

  #if CLA_STAGE_DENSE
    Cla1ForceTask3andWait();
  #else
    dense_fc_cpu(&workspace[POOL1_SIZE], weight, fc_bias, fResult);
  #endif

    // Bu modda her asama zaten bitmis durumda; tek gorev modundaki ISR
    // yerine bayragi burada set ediyoruz ki cagiran tarafin
    // "while(!cla_done)" bekleyisi ayni sekilde calissin.
    cla_done = 1;
#endif
}
//
////
// CLA_configClaMemory - Configure the CLA memory
//
void CLA_configClaMemory(void)
{
    extern uint32_t Cla1funcsRunStart, Cla1funcsLoadStart, Cla1funcsLoadSize;
    extern uint32_t Cla1DataRunStart, Cla1DataLoadStart, Cla1DataLoadSize;
    EALLOW;

#ifdef _FLASH
    //
    // Copy over code from FLASH to RAM
    //
    memcpy((uint32_t *)&Cla1funcsRunStart, (uint32_t *)&Cla1funcsLoadStart,
           (uint32_t)&Cla1funcsLoadSize);

    // DUZELTME (soguk acilis): CLA'nin okudugu const agirliklar (CLADataLS1:
    // weight[], conv agirlik/bias, fc_bias) da flash'tan RAM'e kopyalanmali --
    // eskiden bunu sadece debugger yapiyordu, kart kendi basina acilinca CLA
    // sifir/copp agirliklarla calisiyordu (bkz. linker cmd'deki not). Bu
    // fonksiyon herhangi bir CLA gorevi tetiklenmeden once cagriliyor.
    memcpy((uint32_t *)&Cla1DataRunStart, (uint32_t *)&Cla1DataLoadStart,
           (uint32_t)&Cla1DataLoadSize);
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
    // GUNCELLEME (asama bazli bolusum): yalnizca DERLENEN gorevlerin vektoru
    // yazilir. Tek gorev modunda (uc asama da CLA'da) sadece MVECT1; karma
    // dagilimda asama basina MVECT1=S1, MVECT2=S2, MVECT3=S3. Secilmeyen
    // asamanin gorevi hic derlenmedigi icin adresi de alinamaz -- bu yuzden
    // kosullu (bkz. dense_layer.cla).
#if CLA_SINGLE_TASK
    Cla1Regs.MVECT1 = (uint16_t)(&Cla1Task1);
#else
  #if CLA_STAGE_CONV1_POOL1
    Cla1Regs.MVECT1 = (uint16_t)(&Cla1Task1);
  #endif
  #if CLA_STAGE_CONV2_POOL2
    Cla1Regs.MVECT2 = (uint16_t)(&Cla1Task2);
  #endif
  #if CLA_STAGE_DENSE
    Cla1Regs.MVECT3 = (uint16_t)(&Cla1Task3);
  #endif
#endif

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
    // EKLENDI: Cla1Task1 (conv1+conv2+FC, tamami) bitince buraya dusuluyor --
    // main loop artik bunu bekleyebiliyor.
    cla_done = 1;

    PieCtrlRegs.PIEACK.all = M_INT11;


}
