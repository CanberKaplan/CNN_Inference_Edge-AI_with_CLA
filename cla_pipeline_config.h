#ifndef CLA_PIPELINE_CONFIG_H
#define CLA_PIPELINE_CONFIG_H

// ADOT GUI tarafindan uretildi. 1 = asama CLA'da,
// 0 = asama C28x CPU'da (convolution.c'deki ikizi calisir --
// matematik birebir ayni, sonuc degismez).
//
//   MFCC : HER ZAMAN CPU (mfcc_extract.c) -- CLA'nin 2KB program
//          RAM'ine FFT/log/DCT sigmaz, offload edilebilir degil.
//   S1   : conv1 + relu + pool1 (fused)
//   S2   : conv2 + relu + pool2 (fused)
//   S3   : dense / FC
//
// BELLEK: CLA program RAM'i (RAMLS5_PROG) 2048 byte ve uc asama
// birlikte 2042 byte tutuyor -- bu yuzden ucu de 1 ise hepsi TEK
// gorevde toplanir (CLA_SINGLE_TASK). Biri CPU'ya alininca o kod
// hic derlenmez, acilan yerle asama basina ayri gorev kullanilir.
// CLA veri RAM'i (RAMLS_0_1_2_3_4) 10240 byte; CPU'ya alinan
// asamanin agirlik/tampon kopyasi bu havuzdan cikar.
//
// --- ZAMAN BUTCESI (GUI cevrim modeli, olculmus degil) ------------
// Hedef: TMS320F28379D  (C28x 200 MHz, CLA 200 MHz)
// Ozellik : 200.00 ms (OLCULEN)
// MAC basi 133 cevrim (CLA) / 71 (C28x) -- model varsayimi
// S1   : 40.211 ms  (CLA, 55,800 MAC, 8042 kcevrim) -- diger tarafta 23.669 ms
// S2   : 131.273 ms  (CLA, 201,600 MAC, 26255 kcevrim) -- diger tarafta 67.549 ms
// S3   : 1.508 ms  (CLA, 3,360 MAC, 302 kcevrim) -- diger tarafta 0.905 ms
// Gecikme 372.99 ms, pencere basi 200.00 ms (CLA 172.993 ms'i ortuyor)
// Features 200.0 ms - inference 172.99 ms - latency 373.0 ms, a new window every 200.0 ms, with 172.99 ms of it overlapped by the CLA.
#define CLA_STAGE_CONV1_POOL1  1
#define CLA_STAGE_CONV2_POOL2  1
#define CLA_STAGE_DENSE        1

// --- turetilmis yardimcilar (elle degistirmeyin) ---------------
#define CLA_SINGLE_TASK  (CLA_STAGE_CONV1_POOL1 && CLA_STAGE_CONV2_POOL2 && CLA_STAGE_DENSE)
#define CLA_ANY_STAGE    (CLA_STAGE_CONV1_POOL1 || CLA_STAGE_CONV2_POOL2 || CLA_STAGE_DENSE)
#define CLA_TOUCHES_WORKSPACE  CLA_ANY_STAGE

#endif // CLA_PIPELINE_CONFIG_H
