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
#define CLA_STAGE_CONV1_POOL1  0
#define CLA_STAGE_CONV2_POOL2  0
#define CLA_STAGE_DENSE        0

// --- turetilmis yardimcilar (elle degistirmeyin) ---------------
#define CLA_SINGLE_TASK  (CLA_STAGE_CONV1_POOL1 && CLA_STAGE_CONV2_POOL2 && CLA_STAGE_DENSE)
#define CLA_ANY_STAGE    (CLA_STAGE_CONV1_POOL1 || CLA_STAGE_CONV2_POOL2 || CLA_STAGE_DENSE)
#define CLA_TOUCHES_WORKSPACE  CLA_ANY_STAGE

#endif // CLA_PIPELINE_CONFIG_H
