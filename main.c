

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
#pragma DATA_SECTION(weight,"CLADataLS1");
const int16_t weight [DENSE_LAYER_WEIGHTS] = {
                                            4510, 5058, 8426, -3293, -3970, -9497, -10983, -9401, 6168, 15757,
                                            8619, -9691, 7867, -5653, -7912, -11825, 2455, 10843, 6193, 8812,
                                            7137, -6814, -2678, -11768, 3923, -1232, 4978, 8453, 9481, -1821,
                                            -554, 1886, 18597, 6295, 7960, -1419, -7011, -6655, -3992, 1607,
                                            10339, 10371, 26435, -1311, -8689, 1139, 8055, 2266, 14809, 13782,
                                            -1493, -9121, 22, -4126, 2503, -14162, 781, 10547, 15345, -7506,
                                            5436, -8269, 12004, -7717, 10489, 9864, 610, -13494, -2914, -3093,
                                            3146, -2901, 9148, 157, 408, -8539, 9622, 1571, 4602, -8811,
                                            12557, 4424, 6377, -6235, -1407, -3341, 3055, 1861, 5970, 7903,
                                            5378, -10635, 276, -1576, 11451, -6401, 5277, 20388, 9349, -5621,
                                            -2418, -2202, 6754, -5645, 10349, 1521, 3384, -13787, -9546, 4739,
                                            17482, -6879, -11481, 11526, 11161, -14227, 2009, -4437, 1599, -8206,
                                            -1762, 18485, 10681, -4479, 12533, -5133, 13597, -1629, -1637, 7645,
                                            7366, -16145, 9500, -1998, 13763, -2214, -5771, 12795, 12738, -4053,
                                            5680, -233, 13582, -4282, -8100, 3119, 3041, -6252, 9227, -3018,
                                            15347, -7356, -8287, 8396, 4272, -8770, -3086, -5554, 12716, -10294,
                                            -6146, 11506, 13700, -5444, 839, -9910, 15749, 417, -10786, 2864,
                                            10193, -13076, 33, -10029, 4121, -8633, -2265, 14642, 2130, -12645,
                                            2998, -5774, 16142, -8016, -5978, 17333, 9087, -9722, 5003, -8072,
                                            3124, 822, -9882, 13601, 10806, -3629, 6533, -9705, 4260, -5881,
                                            -11733, 14870, 10152, -13370, -3576, 832, 16886, 2448, -6206, 5317,
                                            5953, -9649, 3857, -9305, 12595, -3922, -6090, 15921, 16357, -2685,
                                            2906, 366, 4037, -4350, -2539, -11181, -15504, 3639, 1506, 9902,
                                            -13412, -2076, -4383, -8314, -17762, 6507, -12099, 3949, -8018, 6262,
                                            -7525, -10064, -6315, 8883, 1732, 17024, -3271, -9673, -6982, -5173,
                                            -13487, 12891, -902, 6656, -9493, -5763, 1297, -5906, -7228, 7727,
                                            2261, 10753, -3694, 4975, -7636, -9735, -16598, 11424, -312, 15978,
                                            -3835, -791, -2441, -4835, -15325, 1420, 4856, 18111, -3009, -3218,
                                            -2914, -10215, 127, 11873, -6645, 6869, -10273, -1727, -10735, -139,
                                            -4336, 7274, -6201, 15328, -11638, -1077, -1502, -4167, -12682, -781,
                                            -9368, 8396, -10317, 3534, -2229, -6064, -15219, 615, 987, 7946,
                                            -15835, 1638, -4482, 4702, -3349, 9456, -1821, 7793, -12443, -6712,
                                            744, 3165, -13294, 11677, -7846, 9507, -13282, 507, -3216, 1349,
                                            -2967, 10824, 3885, 10976, -12894, -5242, 7921, 5537, -3010, 2947,
                                            171, -2036, -10299, -3468, 3801, 1923, -5516, 3709, -4617, -2226,
                                            -7348, 3546, 15627, 1293, -3416, -4897, -11162, -3880, -2556, 4761,
                                            10248, 11812, -771, -4113, 1057, 9312, 417, -3310, 6733, 5565,
                                            506, -5634, -7158, 7998, -1092, 1447, 11200, 7826, -5952, -1431,
                                            -8545, 5361, 2797, 7079, 4968, 6128, -4533, 7662, -7801, 6535,
                                            4091, 7066, 8252, 2401, -5876, 7476, -2539, 4766, 2324, 8292,
                                            4552, 3225, -7484, 2703, -5951, 9840, 717, -62, 14991, 4794,
                                            27, 2209, 6893, 2483, 1913, 2703, 7235, 14519, 562, 3557,
                                            -2505, 687, 667, 3566, 8554, 13340, -3655, 5070, 2217, 10081,
                                            -4688, 3704, 7407, 1753, -903, -8697, -3860, 14698, 879, 8983,
                                            2410, 9381, 2362, 4585, -432, 12842, -3682, 8096, -3269, 2217,
                                            -3373, -4429, -803, 4735, -1913, -11568, -6353, -7248, -10886, 8643,
                                            10542, 14307, -1396, -9262, -9893, 2705, -6730, 3820, 1111, 12775,
                                            -3146, 131, 5451, 8436, -12365, -1805, 851, 10169, -12930, -3513,
                                            5109, 1177, -2014, 7552, 14729, 17469, -10340, -10486, -6233, -1420,
                                            -6685, 4434, 15780, 15413, -13180, 1701, 6809, 1606, -3793, 8333,
                                            12102, 11454, -16563, -2633, -8550, 2422, -7828, 4094, 4355, 6454,
                                            -12653, -13047, -2236, -6757, -1018, 4725, 9258, 11273, -6988, -298,
                                            -3182, 2893, -3335, -2329, 1454, 1309, -13796, -2132, -2206, 2450,
                                            -3875, 3355, 2481, 2398, -14456, -11056, -2782, 3131, -3476, 4873,
                                            10073, 4226, -10693, -2032, 5804, 3988, -12499, -534, 3217, 9993,
                                            -8453, -11913, -9691, -6429, -10311, -1666, 8989, 4879, -16772, -1698,
                                            4419, -863, 7350, 3630, -4183, -1253, -404, 3064, -4866, -14013,
                                            3890, 58, -2936, 4184, -2563, 6436, 3594, -14586, 10628, 3258,
                                            4087, -7463, 1898, 432, -2785, -1484, 10652, 5264, 4766, 1605,
                                            -5004, 5027, 295, -1057, 12197, 9891, 2203, -7790, -5635, 6038,
                                            6564, -8593, 4015, 12686, -3647, -4892, 1441, 12377, 4817, -2718,
                                            66, 9674, -4492, -9268, 5341, 13546, 6228, -6381, 11797, 1047,
                                            -3929, -3265, -2696, 4591, -1095, -1235, 10759, -3395, 4033, 5590,
                                            1718, 4022, -710, -11378, 3900, -212, -7394, -7622, -4716, 2032,
                                            -5242, 533, 12302, 7820, 1745, -3917, 5008, 790, 1044, -5350,
                                            13584, 5521, -2142, -225, 4351, 7431, -5540, -5768, 4891, 783,
                                            -1480, 2298, 6016, 7380, 1870, -12040, 1454, 6974, 3958, -7959,
                                            3491, 6604, 12436, 12993, -3655, -9831, -18319, -3974, -3354, -4313,
                                            25620, 13886, -20770, -8004, -12829, 468, -10581, -9806, 13287, 16402,
                                            -12844, 6270, -16988, 11227, -1255, -7979, 15249, 12066, -11942, 3339,
                                            -12403, -2330, -1221, -6671, 13471, 17859, -8459, -7011, -13194, 1979,
                                            -5427, 382, 10315, 8642, -17806, -3999, -13008, 3892, -10612, -1272,
                                            13469, 10245, -23621, 4427, -13694, 13118, -16021, -546, 19013, 5083,
                                            -24442, -5135, -16551, 13072, -10035, -387, 17542, 9232, -25398, -3250,
                                            -17733, 4368, -5462, -15622, 13498, 524, -13994, -17727, -16706, 566,
                                            -9930, 681, 26983, 3718, -17017, -10479, -16767, 7360, -9435, -838,
                                            20872, 12597, -18047, 5146, -6514, 733, -9550, -10604, 19734, 11590,
                                            -14206, -7490, -24170, 8932, -203, 4847, 13548, 12933, -15188, -2677,
                                            -18060, 9531, -15124, 4058, 3055, -3960, -3491, 4673, -10849, 7593,
                                            -5804, -2161, 12346, 7976, -7581, 2212, -14068, 9577, -16655, -2916,
                                            11150, 819, -13221, 10789, -6113, -269, -6464, 11683, 5722, 8269,
                                            -12223, -2571, -3044, 6520, -6826, 6221, -570, 2795, -10645, 4337,
                                            -10814, 9702, -14187, 9980, -1158, 4459, -8507, 6823, -9950, -1354,
                                            -4533, 5899, -371, 4976, -14744, 2938, -4583, 936, -14720, 4320,
                                            9439, 9672, -13302, 9333, -11114, 3716, -7258, 9627, 4025, -3035,
                                            -9187, 4099, -2295, 4962, -6956, 3830, 8092, 8842, -3480, 11174,
                                            -3162, 468, -3150, 8279, 9142, -14, -4119, 2329, -5527, 6656,
                                            -9355, 112, 11798, 3540, -14624, 4350, 0, -554, -14325, 8771,
                                            4788, 6906, -12340, -752, -177, 11926, -5491, 5537, 6086, -4629,
                                            -13448, 6276, -2544, 10279, -5755, 8148, -10029, -10603, -1196, -8477,
                                            5642, -15384, 5551, -935, -7925, -11544, -2644, -10595, 816, -6484,
                                            6716, -2237, -9186, -9213, 4141, -6586, 9138, -12697, 3802, -7228,
                                            -8873, -10187, -5858, -3897, 233, -11751, 79, -12952, -15322, -4417,
                                            2520, -1873, 8351, -4400, 5455, -2054, -12228, -9309, -6340, -14879,
                                            -695, -13352, 9050, -10761, -7831, -12167, -1806, -9769, -1758, -5856,
                                            5675, -1967, -14895, -14503, -1616, -10270, 1729, -3496, -1530, -6169,
                                            -10842, -3790, -359, -12045, 3716, -9161, 2368, -3110, -9196, -7615,
                                            4125, -1837, 11311, -8131, 2106, -1134, -8583, -2855, -1706, -11584,
                                            -1013, -12050, -1125, -7479, -10929, -2372, 6008, -10913, 1387, -2044,
                                            9497, -10907, -8658, -5056, 1765, -6955, 1204, -15103, 1255, -2337,
                                            -12192, -14614, -2713, -14313, 7575, -10437, 7049, -15709, 7619, 5630,
                                            -3615, 3621, 6641, 84, 734, -3599, -733, -5922, 559, 3354,
                                            8419, -1289, -223, 2775, 9808, 1484, -1431, -6310, -3364, -2098,
                                            -3801, 3948, -1922, -7666, -6776, -3290, 5616, -8743, 7032, -2688,
                                            -2189, -1972, 4018, -5264, 461, -1817, 6465, 1002, 613, 2769,
                                            1787, 3412, -2573, 2866, 1904, 4965, -1592, 3632, -4296, -5125,
                                            -5036, 2037, -3580, 1697, 5607, -4477, -3992, 776, 3123, -5526,
                                            2677, 2707, 3228, -502, 3016, -5532, -2298, -2779, 2195, -2302,
                                            2341, 5433, -389, 3524, 2327, -1066, 834, 3152, 7124, -3969,
                                            -6594, -2425, 1407, 2509, -3932, -5908, -1474, -2670, -4556, -693,
                                            5529, -8165, -2511, 1645, 4773, 2625, -4428, 4714, -1314, -115,
                                            3362, 5833, 8460, 152, -1121, -1853, -3735, 1709, 6510, 810,
                                            -7004, -7326, 3710, 1237, -2543, -10076, -5629, -8452, -3887, -2485,
                                            -2277, -13052, 1352, -9225, 1010, -3395, -6110, -5877, -993, 910,
                                            1308, -7500, 1479, -4871, -2427, -10565, 1842, -8382, -1678, -4381,
                                            6532, 3125, -8316, -4148, 4108, -1132, -36, -3507, -3043, -6177,
                                            -3379, -5804, 5275, -7442, 6707, -13280, 3855, -1462, -11965, -11042,
                                            4792, 2523, -6560, -14029, -5304, -1209, -9662, -12086, 904, -4163,
                                            -1968, -10219, 3072, 5626, -14718, -5024, 3469, -8336, -7089, -575,
                                            3918, -5391, -9474, -13085, 7768, -7044, -6091, -14546, 1669, 2062,
                                            -9403, -12139, -2740, -10306, 7546, -4098, 5885, 5626, -4774, -2839,
                                            -24, -8946, -6333, -8379, 3669, 4195, -11446, -7235, -1160, -3413,
                                            -4288, -5544, 4227, 2620, -10813, -11348, 1862, -9372, 1572, -163,
                                            -441, -8382, -7785, 406, -8675, -1124, -10711, 1749, -6790, -7487,
                                            -3560, 3299, -1518, -8964, 1902, 6436, 2222, 579, -6326, -181,
                                            -7092, -6921, -3204, 5923, -8361, 1722, 1844, 4866, -9426, -2122,
                                            -11343, 2415, -4150, -12364, -3171, 8279, -3979, 2229, -1273, -2613,
                                            5251, -2457, -598, 3500, -5463, 3196, 1866, 6129, 1425, -10405,
                                            -1007, 8662, -4276, -6972, -8053, -2671, -2293, -5188, 4327, 933,
                                            -7575, -8466, -6216, 1173, 3504, -1571, -3185, -2554, -11671, -2132,
                                            -10221, 1276, -5195, -4265, -4900, 9492, -13271, 2157, -6224, -4266,
                                            456, -9114, 2934, 4906, -10076, -4640, -12486, 5593, 6739, -5422,
                                            -2649, 11686, -9684, -4325, -3051, -5330, -5490, -4909, 1565, 5243,
                                            -5437, -5228, -10619, 4224, 6790, -6159, -4863, -14, -12063, -683,
                                            -9993, -5696, -5949, -2359, -18260, -20635, -3977, 10340, 3017, 12825,
                                            -1123, 10437, -17657, -17965, -13513, 7969, -2998, -424, 4318, 8392,
                                            -28019, -13831, -14706, -3184, -6549, 7025, 2671, 17724, -17377, 3950,
                                            -6012, -10140, -8028, 2913, -2470, 2064, -20292, -9706, -6203, 1973,
                                            6988, 9395, -3497, -2203, -21696, -20258, -18089, -2176, 4766, 10019,
                                            -6068, -1728, -29807, -10288, 7228, -5399, 1869, -2625, 2884, 4278,
                                            -28834, -14360, -17803, 5209, -6140, 14161, -3524, 11338, -23607, -15741,
                                            4192, 14761, -1993, -2448, 3587, -2459, -24174, -11454, -6931, 19143,
                                            -5014, 8240, -7890, 19301, -20345, -10657, -15069, 11906, 157, 8119,
                                            -13058, 236, -25818, -8011, -9889, 3806, 1750, -791, -15256, 4118,
                                            -32767, -15152, -21273, 7910, 10167, 5437, -14685, 1981, -24549, -11236,
                                            -8151, 9152, 5376, -2190, -11059, -4872, 2077, -20708, -2465, 8575,
                                            3682, -1253, -7164, -2622, -2053, -11169, 1280, 3574, -4099, -4612,
                                            -8175, -6815, -738, -12018, -5858, 3263, -6459, -5419, 1049, 2683,
                                            -9433, -13052, -15400, 3671, 2201, 4683, -2875, -6714, -5558, -4948,
                                            -3747, 3237, -470, -6256, -1133, 2105, -3333, -13749, -9043, 12193,
                                            -1475, -9055, -9003, 63, -9756, -17771, -6468, -1664, -5460, -4079,
                                            -14699, -3982, -9116, -13584, -9076, -37, 3850, 1622, -7116, -1354,
                                            -12785, -14182, 627, 3361, -4452, 2715, -1873, -8281, -9181, -5295,
                                            -6669, 1975, -5445, 1586, 876, 3118, -13096, -9934, 432, -2010,
                                            -6503, -4346, -211, -9249, -11311, -11759, -3147, -1498, -4831, -6522,
                                            -9522, -6717, -470, -16013, -4830, 9245, -5490, -5012, 1114, -6728,
                                            -7323, -11862, -6392, 6983, -89, -7461, -1200, -4419, 16403, 10394,
                                            7563, 3605, -2421, 2734, 7470, 4691, 15071, 13088, 15743, 2484,
                                            5994, -4448, 7438, 11542, 8235, 7307, 14150, 10663, -92, 4499,
                                            10749, 3899, 6040, 9877, 5511, 2843, -1460, 4341, 1047, 4404,
                                            14782, 12855, 6826, 9208, -2128, 1144, 5345, 5401, 6759, 10089,
                                            7633, 7925, -8126, -2943, 5890, 10179, 14448, 13659, 1452, 11871,
                                            -6068, -4432, 10247, 2261, 13735, 11167, 2983, 4284, -2313, -1128,
                                            8810, 13077, 9927, 14157, 4823, 8486, 826, 6511, 3754, 5194,
                                            11000, 18811, 8246, 8133, -1613, 8379, 4657, 2840, 14469, 9990,
                                            8934, 9956, 3275, -1615, 7142, 123, 3071, 11172, 2327, 12127,
                                            -1674, -2888, 3110, 3505, 8641, 14960, 13077, 10666, -2673, 1746,
                                            12567, 13417, 5677, 5512, 12974, 7008, -7315, -5827, 3899, 10811,
                                            -4864, -17087, 228, -7070, 5057, 4349, 2985, -10314, -18892, -15548,
                                            -172, 6314, 4830, 3783, 8344, -854, -10073, -7789, 6121, 5004,
                                            -2853, -4340, 1564, -12640, -7986, -5742, -2622, 4490, -8204, -6385,
                                            -1391, -5108, -12690, -5400, 1518, -6097, -7891, -6063, -5594, -11205,
                                            -12577, -7680, 8270, 7464, -1104, -5237, 3284, -2448, -10239, -11646,
                                            -2551, 1684, 4056, 1097, 3463, -7972, -11078, -3059, -3188, -4888,
                                            -6267, -211, -2739, -3214, -15847, -4364, -1441, -4307, -7626, -1700,
                                            2173, -299, -6178, -11696, -3506, 6555, -4396, -9363, 1123, -1974,
                                            -3798, -10491, -5356, 3535, 5558, -401, -1740, -8066, -14764, -15408,
                                            4672, -5353, -513, 1198, 3362, -4917, -14774, -15772, 6556, 6709,
                                            4947, -9429, 4080, -4638, -16711, -9406, 53, -937, 3064, -6646,
                                            -8407, -15133, 12478, 2076, 11568, 5189, -8425, 5581, 6852, 7124,
                                            6171, 3459, 9292, 1419, -11012, -7574, 843, 6839, 6763, 6558,
                                            2572, 3616, -8869, -5867, 8716, 3501, 11079, 2807, -1308, 7707,
                                            -4700, -944, 1380, 12198, 16350, 4663, -1280, 9917, -13083, -3886,
                                            11116, 1557, 2971, 12135, 10128, 8159, -15007, -3844, 7445, 7613,
                                            9075, 5539, 1183, 505, -13726, -2504, 10377, 3079, 3800, 6873,
                                            -1841, 4316, -2931, -3487, 15412, 10141, 10923, 990, 331, 7813,
                                            -14240, -368, 12481, 6717, 15297, 6057, 1417, -2362, -2988, 421,
                                            10867, -854, 12403, 8977, 4075, 1043, -14661, -7430, 1746, 2094,
                                            7463, 4888, 639, -3742, -2892, -4472, 2453, 12026, 16488, 10683,
                                            12076, -6265, -3335, -7606, 11839, 4515, 13440, 6810, 4333, 2449,
                                            -9449, -8038, 5000, 3685, 59, 4, -3912, -1468, -2403, -7041,
                                            3285, -5107, 6692, -665, -6044, -2572, -963, 2760, -4083, -6292,
                                            11009, 6509, 2743, -532, 6547, -6738, 851, -4671, 11715, 3762,
                                            1251, -8771, 6160, -5204, 812, -157, -2078, -4350, 6060, 2809,
                                            5847, 4015, 6409, -181, 7082, -2379, 810, -10788, 4696, -3136,
                                            -4572, -6455, 1892, 7848, -4076, -5670, 7177, -1431, 343, -3950,
                                            -2112, 5658, -5297, -4690, 5239, -4428, -1577, 3666, 4744, 18,
                                            3226, -7447, 2252, -2389, -1860, 5343, 1591, 5447, 6756, -6007,
                                            1519, 443, -3774, -2872, -2131, 5259, -1941, -7094, 8434, -4418,
                                            -866, -4738, 3773, -870, 5452, -6521, 6488, -1606, -4349, -7375,
                                            8088, 2385, 84, 1739, 5376, 8514, -3159, -1048, 8589, 7054,
                                            -4282, -5737, 6528, 3660, -6820, -5087
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
volatile uint16_t g_predicted_class = 0;
volatile uint16_t g_correct_count = 0;
volatile uint16_t g_total_count = 0;

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

        GPIO_WritePin(19, 0);

        // Tum inference (conv1+conv2+FC) tek CLA gorevinde -- burada,
        // sonucu DOGRULAMAK icin bilerek bekliyoruz (cla_done), aksi halde
        // hangi girisin hangi sonucu urettigini eslestiremeyiz. Bu,
        // "CPU'yu bir sonraki pencereyi hazirlamak icin serbest birak"
        // ilkesiyle celismiyor -- gercekten sonuca ihtiyac duydugumuz an
        // (burada: dogrulama) beklemek dogru olan.
        cla_done = 0;
        CLA_runTest();
        while (!cla_done) { /* CLA'nin bitmesini bekle */ }
        GPIO_WritePin(19, 1);

        // argmax(fResult) -> tahmin edilen sinif
        {
            uint16_t best_k = 0;
            for (k = 1; k < NUM_CLASSES; k++) {
                if (fResult[k] > fResult[best_k]) best_k = k;
            }
            g_predicted_class = best_k;
            g_total_count++;
            if (g_predicted_class == g_expected_class) g_correct_count++;
        }

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
    // EKLENDI: Cla1Task1 (conv1+conv2+FC, tamami) bitince buraya dusuluyor --
    // main loop artik bunu bekleyebiliyor.
    cla_done = 1;

    PieCtrlRegs.PIEACK.all = M_INT11;


}
