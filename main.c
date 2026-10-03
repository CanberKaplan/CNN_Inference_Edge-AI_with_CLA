

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
// GUNCELLEME (MIMII): golden vector seti modelin sinif sayisina gore
// seciliyor -- 2 sinif = MIMII fan (normal/abnormal), 3 sinif = BLDC.
// ADOT her export'ta NUM_CLASSES'i definitions.h'e yaziyor, yani hangi
// veri setiyle egitilen model port edilirse edilsin dogru set derlenir.
#if NUM_CLASSES == 2
#include "golden_vector_normal.h"
#include "golden_vector_abnormal.h"
#elif NUM_CLASSES == 3
#include "golden_vector_healthy.h"
#include "golden_vector_bearing.h"
#include "golden_vector_propeller.h"
#else
#error "Bu sinif sayisi icin golden vector seti yok (2 = MIMII, 3 = BLDC)"
#endif
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
    -2806, 8254, -245, 18329, 12395, 6513, 15140, 7821, 3551, 3433,
    3896, 13325, 2460, -356, 5074, 6189, 11454, -801, 527, -7299,
    -4297, -870, 14766, 13009, 13228, -9888, -45, 6106, -2037, 5249,
    1606, -4870, 3556, -5638, 3116, 16293, -11673, -17640, 7295, 5453,
    2665, -1644, -1266, 11654, 10667, -939, 10945, 10540, -1903, -10406,
    -2542, 2111, 397, -305, -5650, -3593, -6448, 6486, -1012, 5201,
    11097, 13537, 13558, 9591, -589, 4279, 1990, 17891, 4002, -7348,
    12927, 1956, -3438, 1741, 6140, 2066, -2006, 3149, 10332, 3950,
    6704, -586, 2886, 1313, 1992, 4564, 7699, 19373, 7855, -6694,
    -5532, -2026, -11862, -4171, 5826, 270, 2381, -10751, -823, 1893,
    -6196, -1652, 9787, 2326, 5142, -4967, 6745, 12344, 10230, 5170,
    3065, 7824, -3966, -8318, -22156, 11586, -9597, 4814, -3490, -3662,
    -6276, 2543, -10177, -1248, -1426, 11066, -2876, -8729, -226, -2999,
    -21651, -156, -3683, 7300, -5945, -9967, -4899, -2012, -19073, -5517,
    -7003, 4278, -2365, 2884, -2605, -12823, -11598, 7489, 2732, 582,
    -8205, 733, -706, -1189, -3262, 7791, 3634, 3392, 3198, 3379,
    198, 1529, -12624, 1751, -5601, 9108, -1681, -5995, -178, 2731,
    -5556, 2252, -3691, 7296, -7544, -12188, -4977, -2571, -11821, 9784,
    323, 10512, -4523, -4325, -10133, -5211, -4761, 2881, -2594, 9231,
    2759, -1004, -7219, 3538, -16003, -1653, -4888, 7769, -3511, -9871,
    -4615, -6881, -19971, 3168, 1631, 7117, -4218, -5302, 1018, -10882,
    -4823, 7836, -4784, 3678, -11366, -4104, -1889, -5444, -8369, 11986,
    -3704, -784, -2363, 745, 4856, -3973, -6483, 13146, -15666, 8706,
    -11301, 5105, 3926, -3674, -2772, 14229, -9128, -2979, -338, 5201,
    4357, -1823, -1715, 13856, 427, 3953, 829, 9311, -1948, 2310,
    -6057, 10165, -6649, 10414, 2041, 7023, 1131, -2548, -4813, 10434,
    -12153, 6010, -13735, 3958, 3322, -184, -4679, 7795, -20876, 5931,
    -14741, 8015, -2148, 814, -2690, 7213, -26589, 1695, -11038, 3959,
    -3284, -148, -3783, 13423, -13317, -1468, 5882, 12305, 1677, -4168,
    -3917, 13610, -14326, 2404, 2847, 6605, 5339, -2910, -1253, 12102,
    -9483, 1055, -1995, 1734, 2573, -3861, -135, 7795, 999, 6272,
    -2071, 5203, 1004, -760, -4106, 5206, 469, 1253, 11548, 8806,
    2090, -1718, -3495, 12105, -23127, 4740, -14944, 10271, -1616, -3574,
    -10338, 14391, -24860, 14816, -6660, 10971, -2104, -9165, 3853, 3272,
    16743, 4739, 1566, 12434, -9276, -9068, -2237, 3115, 6468, -3179,
    -3226, 1465, -1249, -11871, 1131, 2018, 8525, -2714, 2729, 6702,
    -2784, -2346, -2361, -3997, 6226, 4529, 6245, 9186, -2612, -16298,
    -7707, 2049, 3520, -4743, 20282, 3067, -3869, 1159, 2356, 1804,
    16893, -11670, 10107, -4069, -5710, -4491, -4136, -5942, 11536, 6565,
    5589, 4244, -3864, -7790, 5379, 3799, 11743, 989, 4287, 7327,
    -4280, -5733, -4745, 11769, 13769, 4475, 4375, 8249, -839, -11205,
    1529, 4790, 5924, 5876, -2914, 4029, -4039, -16440, -10395, -574,
    13528, 12462, 1848, 5091, 1116, -2526, -1017, -9486, 11975, 4847,
    1637, 10775, -5636, -6945, -5604, 1499, 1393, -982, 19668, 4665,
    535, -7834, 3200, 8387, 22348, -4889, 17369, -4884, 6668, -4114,
    620, -6725, -5642, -3723, -14940, 1538, 5498, -6597, -6895, -11849,
    -11370, -2758, -5666, -1965, 3508, -2485, 384, -5257, 1607, -4052,
    -15429, 8092, -205, -3442, -3175, -2258, 5472, 1386, -7130, 5956,
    3668, -5914, 1438, -3428, 9012, -1967, -10571, 2928, -1657, 3205,
    507, 2336, -1094, -8770, -18282, 10689, 8343, 1006, -5097, -2417,
    3218, -8883, -28714, -1440, 8703, 2983, -6724, -4427, 544, -5936,
    -9316, 1950, 9245, -5343, -744, -2902, 3760, 5537, -12439, 5699,
    -1782, -6000, -3854, -10475, -3149, 284, -4698, -993, 2046, -2252,
    -6901, -5736, 2231, -1236, -18974, 3851, 1429, -6206, -5377, 3618,
    10951, -5287, -13124, 4345, -1398, -6953, -2018, -5403, 16554, -3339,
    -12344, 6347, 8426, -4122, -1413, 1177, 2786, -7753, -22730, 11250,
    5444, -2257, -19063, -1197, -9515, -2880, -8845, -7627, 7167, 8454,
    -1697, 4035, -2813, 4401, -8331, -2857, 865, -1994, -7508, -1630,
    -2682, 991, -16669, -8638, -3991, 2104, -4957, 11087, 2727, -9720,
    -19413, -10088, -5560, 4011, 3609, 4796, -5533, -8864, -5352, -4609,
    -1055, 2749, -6668, 3543, 2672, 8217, -4573, -12223, -7354, -769,
    -192, -586, -9395, 4861, -8746, 6457, -1079, 8996, -9412, 9396,
    -722, -2996, -13315, -5776, 1369, 718, -10386, 2919, -814, 2034,
    -3357, -10065, 8529, 2960, -14272, 303, 787, 4933, -7043, -6692,
    3262, 8089, -619, 9822, -1674, -1179, -11204, -4208, -2330, 346,
    -10246, 4949, 6502, -7334, -21369, 214, -16891, 5520, -723, 7913,
    -10722, -5572, -8933, -3418, -3263, 12232, 6246, 10502, -721, 9568,
    -7991, -9948, 13068, 6412, -12490, -16205, 9111, 905, 16276, 8351,
    7266, -7502, -20027, -19171, 106, -4123, 24702, 699, 7418, 169,
    -7772, -5721, 13568, -7292, 5033, -6542, 1217, 3343, 1217, -2451,
    3067, -7099, 637, -2677, 11449, 6306, -12328, -9296, -5478, -4731,
    8404, 6452, 17646, 2864, -16901, 4372, 23461, 3609, 5679, 2299,
    6223, 17601, -26281, -3997, 2476, -1930, 10246, 6081, 959, -6905,
    -26289, -7624, -5308, -7497, 11723, 9677, 9880, 5440, -10672, -7969,
    9127, -9902, 11214, 2550, 5019, -5037, -17898, -10090, -13998, -2315,
    9887, 6851, 11564, 1999, -13577, -3900, 5588, -7341, 1656, -11767,
    4467, 3698, -9112, -1509, 4676, -13857, 9946, 2857, 7260, 860,
    -16508, -2642, -14971, -15862, 1794, 4363, 18285, 6381, -32767, -9834,
    16955, -5584, 14626, 1559, -875, -5849, 3298, 17882, -3659, 2780,
    8008, 14356, -2146, 618, -799, 3773, -11192, 2938, -2247, 16296,
    -2307, -7212, 3010, 6781, 5554, -1544, -1068, 10262, -2109, -6201,
    10577, 6360, 699, 48, 12228, 11196, 114, -2453, 5474, 2331,
    -7450, -844, 4346, 13796, -3544, -9364, 6853, 4997, -6289, 2990,
    2838, 10293, -1235, -6082, 9469, 3608, -7922, 3278, 6764, 11131,
    -607, -7281, 1391, 10156, -4002, -2481, 329, 9702, -4252, -2299,
    6865, 17531, 11105, 4028, 3177, 8533, 3351, -1696, 90, 1860,
    -12014, 1630, 2940, 18218, -808, -5684, -3227, 1281, -4882, 3124,
    3075, 10313, -804, -5223, 5669, 6579, 11065, -1977, 1541, 8320,
    6042, -58, 4274, 23740, -10875, -4190, 7915, 13971, -2841, -8851,
    15463, 6770, -9521, 564, -1399, 14146, -21350, -2850, 9860, -1127,
    -211, 749, 5492, -6437, -30490, -1437, 3855, -7525, 3777, 3737,
    -4587, 3019, -22608, -9834, 12308, 3468, 4956, 5065, 5907, 3730,
    -10664, 5877, -8174, 585, 7001, 7313, 7260, -1595, -22607, 2497,
    6005, 1905, 6575, -8376, -2502, 4658, -14895, 5493, 7001, 5554,
    3478, 1301, 5307, -1531, -18712, -4694, 395, 2724, 190, 3321,
    1480, -6349, -17333, 2567, 4054, -1000, 972, -6016, 875, -5761,
    -13662, -8930, 1330, -3678, 3129, 2943, 6006, -4664, -25173, -3235,
    2349, 3174, -10890, 5543, 307, 1246, -10546, -10898, 3562, 1814,
    -1861, 4355, -3821, -3887, -16209, 1796, -755, -2091, 5142, 2913,
    201, 5252, -24248, 3077, 3431, 2414, 7865, -3978, 386, 4332,
    -28162, 2986, 6496, 6474, -181, 4992, 11036, 456, 2014, -5435,
    -1815, -17764, -11295, -3867, -14359, -7069, -5043, -4391, -1785, -14768,
    2021, -4, -5310, -8116, -11027, -1920, 289, 5757, 6429, 1819,
    -14367, -13080, -13345, 5778, -752, -4362, 3916, -5731, -6448, 4194,
    -400, 10087, -3054, -14803, 7414, 16412, -8209, -3183, -4820, 1507,
    3679, -9754, -7883, 3142, -8972, -8846, 802, 10067, 2436, -3571,
    71, 3872, 6412, 3619, 2369, -2549, 2047, -8289, -8416, -13190,
    -14704, -9639, -4461, -2153, -5498, -17333, -5173, 7499, -12933, -570,
    1193, -2785, -8321, -5206, 6232, -2185, -10228, -260, -4558, 892,
    -3417, 1916, -1523, -7392, -11697, -19866, -7653, 9699, 3800, 3629,
    15568, 1743, -10189, -2782, -3186, 5951, -2354, -1370, 9245, 299,
    -11777, -5384, -7459, 5087, -1933, -12495, -10524, -5414, -7036, -7727,
    6526, 6069, 19725, -13245, 9225, -6143, 2352, 893, 5766, -86,
    13936, 2873, 426, -12097, 1368, 10011, 1634, 1127, 21855, -1401,
    4314, -9755, 7309, 10445, 5307, 4518, 19584, 1759, 10161, -3022,
    4301, -3718, 5637, 10168, 9855, -5439, -2212, -937, 8267, 1808,
    -2531, -331, 6069, -4408, -864, -1092, -2207, -5307, -2171, -1379,
    14240, -2203, 4785, -6653, -300, 4764, 2048, 244, 8976, 690,
    -1237, -4040, 4932, 8880, 8145, 3262, 15814, -9157, 838, -11761,
    2621, 3615, 9530, 6037, 4789, -265, 3616, -10670, -346, 5091,
    6985, -5402, 16383, 2332, 3838, -7941, 1352, 10044, 2936, 3573,
    19961, -917, 98, -3916, 7338, 3154, 394, 6838, 3125, -6719,
    5352, -1041, 10797, 3540, 3533, 5562, 6638, -9525, 5561, 1879,
    2373, 641, -367, 5598, 6085, -16796, 15565, -7789, 11639, -9390,
    -3117, 4479, 3794, -14965, 10607, -713, 3308, -4581, -1420, 1710,
    1372, -14368, -1295, -1076, -414, -5432, 2423, -1247, 4232, -8086,
    10947, -6277, -4573, -5637, 1279, 1924, 4686, -9153, 13842, -2427,
    11792, -6570, -1702, 1915, 2805, -4477, 22192, -7967, 12783, -9899,
    2171, -1131, 596, -8129, 28235, -940, 13594, -3724, 3446, 2363,
    3288, -12497, 10653, 2921, -4291, -7829, -1248, 3820, 4288, -10958,
    12102, -4768, -5280, -3930, -5296, 6164, 863, -8417, 6438, -744,
    342, -1077, 325, 3666, 878, -11829, -2185, -5984, 45, -6475,
    -2044, 4147, 8760, -6365, -3515, -2480, -9543, -6605, -3192, 3163,
    3195, -12257, 23453, -4713, 17384, -9824, 1239, 2207, 9288, -15351,
    25190, -13921, 8309, -13945, 5846, 12151, -7632, -2516, -17035, -4392,
    1006, -11192, 6334, 9194, 2442, -7559, -4242, 2840, 1784, 2088,
    2212, 10555, -352, -1588, -8099, 1340, -6327, -6839, -1904, 863,
    3373, 2468, -10438, -1587, -5687, -11960, 3809, 18034, 8849, -780,
    -7418, 4561, -17980, -2151, 2742, 1567, -2128, -2282, -20478, 13810,
    -12477, 4976, 5583, 4681, -758, 8629, -10452, -4724, -8655, -6647,
    5512, 8739, -7245, -2925, -15886, -2780, -4898, -8653, 3023, 4096,
    4097, -9575, -17968, -2219, -3429, -9659, 3822, 14063, -1873, -8858,
    -7727, -4544, 2634, -4452, 6235, 16195, 12780, 2529, -14102, -11897,
    1827, -6794, 1759, 1262, 3124, 9237, -11920, -6112, -3203, -12595,
    3266, 4209, 7258, -1517, -5766, -1660, -15143, -7624, -4228, 11573,
    -1294, -5243, -21755, 191, -12567, 4877, -4232, 2896, 572, 3657,
    3107, 4223, 16990, -1703, -6121, 5989, 4906, 10175, 9581, 3697,
    4169, -2970, -2166, 3584, 1034, 2729, -4487, 3000, 12651, -6458,
    -340, 2683, 5440, 3493, -3388, 297, 9972, -9793, -3853, 4296,
    1175, 3480, -11082, 1192, 10786, -5669, 1089, -4829, -4827, -3003,
    3966, 13832, 20740, -7883, -4736, 883, 5104, 169, -5409, 5190,
    26779, -1940, -9141, -5604, 6621, 5663, -2857, 5329, 8873, -2527,
    -7776, 6405, -3067, 2338, -6498, -6746, 12650, -5758, 145, 8369,
    3470, 11892, 3509, -3245, 4179, -42, -2937, 6443, 5768, 7778,
    1491, 2151, 20443, -3568, -3094, 8115, 3090, -88, -11985, 2667,
    12722, -7593, -343, 4843, 1313, 6510, -12382, 1257, 12663, -4792,
    -8068, 319, 2860, 349, -2896, 10186, 22025, -10955, -7068, 4828,
    19643, 1011, 6182, 660, 11063, 8096, -8167, -11366, 1629, -561,
    2100, -2522, 7635, 2526, -1837, -124, 4733, -2956, 4648, -1237,
    20007, 9940, 5860, -2236, 5554, -12254, -3581, 7684, 16193, 12147,
    1622, -5939, -6501, -8053, 5295, 5839, 7955, 4182, 1637, -4291,
    4787, -3717, -1952, -9897, 6312, 12806, 7962, 1759, 1243, 946,
    8463, -3539, 5258, -5074, -331, -7454, 7420, -9026, 1104, 1762,
    8430, 5413, -1252, 388, 10753, -3512, 292, 786, 6121, 9459,
    -10058, -3142, 15508, -1294, -1165, -5856, 7656, 9006, -1486, -6996,
    242, -9571, 1304, 3139, 13510, 5989, 2987, 1278, 12005, -6648,
    -7560, 4414, 22871, 4559, 12901, -8405, -1017, -8687, 8224, 4666,
    9148, 4088, 3090, -13247, -7523, -11235, -1849, -5474, 5041, 10456,
    -10191, -6572, 12110, 16277, -9676, -1930, -16081, -7941, -7549, 6557,
    18841, 20659, 2514, 7270, -25051, -96, -9924, -2409, 9099, 7873,
    -15092, 7312, -3682, 6195, -740, -5261, 1089, 4350, -5251, 7905,
    1404, 3531, -13308, -5759, 13408, 10107, 7446, 2525, -7685, -5386,
    -16838, -1439, 19103, -2581, -22080, -4082, -3353, -2016, -8525, -18202,
    27590, 4450, -3354, 902, -7973, -6316, -1460, 7806, 24134, 6801,
    5140, 12295, -13908, -14241, -10161, -6192, 12320, 5881, -9965, 8151,
    -12137, -5984, -2340, 7387, 13277, 13038, 15745, 3064, -10677, -1986,
    -12221, -4154, 12828, 3771, -2912, 6665, -1873, 7002, -3998, -4013,
    8671, -865, -7720, 9638, -6568, -2843, -7706, 1453, 19575, 5452,
    14953, 12102, -5399, -6769, -14478, -6150, 30423, 10590, -18627, 4057,
    -14354, -1740, 834, 6174, -4233, -19257, 2775, -2679, -5624, -9749,
    2551, -2414, -1776, -7376, 8532, -2157, -385, -18169, -1481, 8878,
    -5533, -5810, -2862, -1015, 1322, -10154, 5745, 9389, -13494, -5620,
    -2195, -4230, -11091, -10683, -4859, 4072, -2940, -776, 4471, 992,
    -7801, -10739, 620, 8154, -7390, -4901, 4662, -232, -608, -11343,
    4568, 7592, -10731, -2066, 5840, -3505, -7292, -7896, 3085, 6046,
    -2002, -7172, 7103, 2614, -793, -10465, 1180, 1844, -5519, -17259,
    -11604, -4040, -2861, -9934, 30, 5148, -2513, -2556, 10576, 525,
    -6359, -16433, -3225, 8183, 2975, 1850, -77, -3825, -1249, -12347,
    2097, 3783, -4042, -6039, -10618, 2648, -4560, -7793, -4690, -2295,
    -5126, -21179, 8566, 2117, -7375, -16595, -597, 8530, -13575, -7112,
    12090, -3079, -2374, -14377, 25410, 2917, -11864, -2338, 574, 897,
    -8957, 3548, 27572, 735, -1594, 3874, -1656, -1580, 7148, -3340,
    20218, 8849, -7790, -5018, -8367, -4406, -5282, -5013, 9924, -4468,
    4630, -809, -7140, -6806, -6303, 2387, 21870, -352, -6138, -356,
    -4109, 7236, 1116, -3013, 15218, -7252, -6524, -3155, -3346, -1024,
    -3818, 4861, 15924, 1171, 1569, -1278, -2114, -3062, -672, 6626,
    13495, -5118, -7203, -28, 826, 4827, 342, 3087, 16321, 10101,
    -2305, 2843, -4011, -3670, -4011, 8884, 26813, 5829, -5672, 384,
    9409, -2501, -1601, -4803, 11820, 11181, -3159, -4618, 2201, -3394,
    2058, 5055, 15209, -918, 2048, 742, -6916, -1364, -2263, -1895,
    21819, -2728, -6849, 74, -7993, 5672, 1861, -7730, 28411, -2835,
    -7873, -5582, -4275, -6919, -6498, 214
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

void CLA_configClaMemory(void);
void CLA_initCpu1Cla1(void);

static float features[FEAT_N_FRAMES][FEAT_N_OUT];   /* 65x12 (eskiden 32x12 -- SLICE_LEN 2x oldu) */
__interrupt void cla1Isr1();

// EKLENDI: golden-vector dongusu -- gercek ADC yerine, uc sinifin
// (Healthy/Bearing/Propeller) onceden hesaplanmis golden_adc dizisi
// sirayla beslenip pipeline'in giris degisimini dogru yakalayip
// yakalamadigi (CLA sonucu -> beklenen sinifla eslesiyor mu) test edilir.
#if NUM_CLASSES == 2
static const uint16_t *const golden_inputs[NUM_CLASSES] = {
    golden_adc_normal, golden_adc_abnormal
};
static const char *const golden_names[NUM_CLASSES] = { "Normal", "Abnormal" };
#else
static const uint16_t *const golden_inputs[NUM_CLASSES] = {
    golden_adc_healthy, golden_adc_bearing, golden_adc_propeller
};
static const char *const golden_names[NUM_CLASSES] = { "Healthy", "Bearing", "Propeller" };
#endif
// CCS'in Expressions/Watch penceresinden debug sirasinda canli izlemek
// icin: hangi giris beslendi, model ne tahmin etti, dogru mu.
volatile uint16_t g_expected_class = 0;
// GERI EKLENDI: inference sonucu. run_inference() her pencereden sonra
// fResult[]'in argmax'ini alip bunlari gunceller:
//   g_predicted_class  modelin karari (0..NUM_CLASSES-1)
//   g_logits[]         o karar icin fResult[]'in kopyasi
//   g_correct_count / g_total_count   golden dongusundeki dogruluk
// CPU'da calisan (ya da karma) dagilimda sonuc run_inference() donunce
// hazirdir. Uc asama da CLA'daysa (CLA_SINGLE_TASK, paralellik modu) CLA
// beklenmez; o zaman bir ONCEKI pencerenin sonucu, CLA bitirmisse, bir
// sonraki tetikten hemen once kaydedilir -- yaris olmadan.
volatile uint16_t g_predicted_class = 0;
volatile float    g_logits[NUM_CLASSES];
volatile uint32_t g_correct_count = 0;
volatile uint32_t g_total_count = 0;
const char *volatile g_predicted_name = 0;

void run_inference(uint16_t expected_class);

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

    // EKLENDI: sirayla beslenecek sinif indeksi (MIMII: 0=Normal,1=Abnormal;
    // BLDC: 0=Healthy,1=Bearing,2=Propeller; sonra basa sarar). static: fonksiyon disina tasinsa da her cagrida degerini korur.
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

        // GPIO19: tamamen CPU'da (ya da karma dagilimda) dusuk kaldigi sure
        // INFERENCE SURESIDIR. Uc asama da CLA'dayken CLA beklenmedigi icin
        // yalnizca tetikleme ani kadar (mikrosaniyeler); o modda inference
        // suresi icin GPIO67'ye bakin -- onu Cla1Task1 kendi icinden suruyor.
        GPIO_WritePin(19, 0);

        // INFERENCE: run_inference() S1 -> S2 -> S3'u cla_pipeline_config.h'a
        // gore CPU'da ya da CLA'da calistirir ve sonucu (g_predicted_class,
        // g_logits, dogruluk sayaclari) kaydeder. Tamamen CPU'daki modelde
        // GPIO19'un dusuk kaldigi sure inference suresinin kendisidir.
        //
        // PARALELLIK MODU (yalnizca uc asama da CLA'dayken): CLA tetiklenir
        // ve BEKLENMEZ. CPU dongunun basina donup bir sonraki pencerenin
        // MFCC'sini hesaplamaya baslar; CLA bu sirada onceki pencerenin agini
        // isletir. Iki pin ayni anda dusuk gorunur:
        //     GPIO18 dusuk = C28x oznitelik cikariyor
        //     GPIO67 dusuk = CLA inference yapiyor   (Cla1Task1 suruyor)
        // Sonuc o modda bir pencere gecikmeli kaydedilir (bkz. run_inference).
        //
        // TEK TAMPON, VE NEDEN YETIYOR: CLA ile CPU'nun paylastigi tek
        // degistirilebilir tampon input_image[]. workspace[] ve fResult[]
        // yalnizca CLA'nin, agirliklar sabit, ve CPU fResult'i yalnizca
        // cla_done 1 iken okuyor.
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
        run_inference(golden_cycle_idx);
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



// fResult[]'in argmax'i -> g_predicted_class, g_logits[], dogruluk sayaclari.
static void record_result(uint16_t expected_class)
{
    uint16_t c, best = 0;
    for (c = 0; c < NUM_CLASSES; c++) {
        g_logits[c] = fResult[c];
        if (fResult[c] > fResult[best]) best = c;
    }
    g_predicted_class = best;
    g_predicted_name = golden_names[best];
    g_total_count++;
    if (best == expected_class) g_correct_count++;
}

// INFERENCE: input_image[] (mfcc_extract'in int16 ciktisi) ->
//   S1 conv1 + relu + pool1 -> S2 conv2 + relu + pool2 -> S3 dense
//   -> fResult[] -> sinif karari (record_result).
//
// Her asama cla_pipeline_config.h'a gore ya convolution.c'deki CPU
// fonksiyonuyla, ya da dense_layer.cla'daki CLA goreviyle kosar. Uc bayrak da 0
// iken (tamamen CPU) butun ag bu fonksiyonun icinde, C28x'te calisir ve
// fonksiyon dondugunde fResult[] hazirdir.
void run_inference(uint16_t expected_class)
{
#if CLA_SINGLE_TASK
    // Uc asama da CLA'da -- paralellik modu: tek tetik, asenkron, CLA
    // beklenmez. Bir onceki tetigin sonucu, CLA bitirdiyse, ait oldugu
    // pencerenin sinifiyla kaydedilir -- yeni tetik fResult'i yeniden
    // yazmadan ONCE.
    static uint16_t pending = 0, pending_class = 0;
    if (pending && cla_done) {
        record_result(pending_class);
    }
    cla_done = 0;
    Cla1ForceTask1();              // conv1+conv2+dense tek CLA gorevinde
    pending = 1;
    pending_class = expected_class;
#else
    cla_done = 0;

    // S1: conv1 + relu + pool1  (input_image -> workspace[0..POOL1_SIZE))
  #if CLA_STAGE_CONV1_POOL1
    Cla1ForceTask1andWait();
  #else
    conv1_pool_fused_cpu(input_image, workspace, conv1_weights, conv1_bias);
  #endif

    // S2: conv2 + relu + pool2  (workspace -> workspace[POOL1_SIZE..])
  #if CLA_STAGE_CONV2_POOL2
    Cla1ForceTask2andWait();
  #else
    conv2_pool_fused_cpu(workspace, &workspace[POOL1_SIZE], conv2_weights, conv2_bias);
  #endif

    // S3: dense / FC  (pool2 ciktisi -> fResult[NUM_CLASSES])
  #if CLA_STAGE_DENSE
    Cla1ForceTask3andWait();
  #else
    dense_fc_cpu(&workspace[POOL1_SIZE], weight, fc_bias, fResult);
  #endif

    cla_done = 1;
    record_result(expected_class);
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
