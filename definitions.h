/*
 * definitions.h
 *
 *  Created on: Feb 11, 2026
 *      Author: canbe
 */

#ifndef DEFINITIONS_H_
#define DEFINITIONS_H_



// GUNCELLEME: Prepare_features.py'de SLICE_LEN 8448->16896'ya (0.528s->1.056s,
// 32->65 cerceve) cikarildi -- dosya-bazli 5-fold CV'de macro-F1'i
// 0.979+-0.029'dan 0.991+-0.011'e iyilestirdigi icin (bkz. egitim tarafi
// notlari). IMAGE_H bu yuzden 32 degil 65. IMAGE_W (MFCC katsayi sayisi)
// degismedi.
#define IMAGE_H              65   // 65 (eskiden 32)
#define IMAGE_W              12   // 12
#define NUM_FILTERS          4  // 6
#define NUM_CLASSES          3  // Healthy=0, Bearing=1, Propeller=2

#define FILTER_SIZE          3  // 3x3

// --- 1. Katman Boyutları (Conv1 -> Pool1) ---
// Valid Convolution: (Girdi - Filtre + 1)
#define CONV1_OUT_H          (IMAGE_H - FILTER_SIZE + 1)  // 30
#define CONV1_OUT_W          (IMAGE_W - FILTER_SIZE + 1)  // 10
// 2x1 Pooling: SADECE yükseklik (H) yarıya iner, Genişlik (W) sabit kalır!
#define POOL1_OUT_H          (CONV1_OUT_H / 2)            // 15
#define POOL1_OUT_W          (CONV1_OUT_W)                // 10

// --- 2. Katman Boyutları (Conv2 -> Pool2) ---
#define CONV2_OUT_H          (POOL1_OUT_H - FILTER_SIZE + 1) // 13
#define CONV2_OUT_W          (POOL1_OUT_W - FILTER_SIZE + 1) // 8
// 2x1 Pooling: SADECE yükseklik (H) yarıya iner
#define POOL2_OUT_H          (CONV2_OUT_H / 2)               // 6
#define POOL2_OUT_W          (CONV2_OUT_W)                   // 8

// --- Bellek (Workspace) Offset Hesaplamaları ---
// Dizilerin kendi üzerine veya birbirinin üstüne (overwrite) yazmasını engellemek için
// her bir çıktının boyutunu tam olarak hesaplıyoruz.
#define CONV1_SIZE           (NUM_FILTERS * CONV1_OUT_H * CONV1_OUT_W) // 6 * 30 * 10 = 1800
#define POOL1_SIZE           (NUM_FILTERS * POOL1_OUT_H * POOL1_OUT_W) // 6 * 15 * 10 = 900
#define CONV2_SIZE           (NUM_FILTERS * CONV2_OUT_H * CONV2_OUT_W) // 6 * 13 * 8  = 624

// Dense katmana giren düzleştirilmiş veri boyutu
// GUNCELLEME: IMAGE_H 32->65 oldugu icin 288->672 (6*14*8), 864->2016 (672*3)
#define DENSE_LAYER_INPUT    (NUM_FILTERS * POOL2_OUT_H * POOL2_OUT_W)  // 672
#define DENSE_LAYER_WEIGHTS  (DENSE_LAYER_INPUT * NUM_CLASSES) // 672*3 = 2016

// GUNCELLEME (DUZELTME): eski formul CONV1_SIZE+POOL1_SIZE+CONV2_SIZE idi --
// bu, conv1/conv2'nin HAVUZLANMAMIS (ham) ciktisini da workspace'te ayri
// ayri saklayan eski (fused OLMAYAN) bir tasarimdan kalmaydi. Gercek
// dense_layer.cla::conv1_pool_fused_cla/conv2_pool_fused_cla fonksiyonlari
// havuzlanmamis ciktiyi HIC saklamiyor (her pooled deger, 2 satirlik
// pencereden dogrudan, ara depolama olmadan hesaplaniyor) -- yani
// workspace'in gercekte ihtiyaci olan tek sey POOL1_SIZE (pool1 ciktisi,
// conv2'nin okudugu) + DENSE_LAYER_INPUT (=POOL2_SIZE, pool2 ciktisi = FC
// girdisi). Eski (yanlis buyutulmus) formul CLA'nin dar RAMLS havuzunda
// ~9000 byte bosa harcatiyordu -- bu da weight[]'in (4032 byte) CLA'da
// kalmasini engelleyip FC'yi CPU'ya tasimamiza yol acmisti. Duzeltilince
// FC yeniden CLA'da kalabiliyor (asagida dense_layer.cla'ya geri tasindi).
#define MAX_ELEMENTS         (POOL1_SIZE + DENSE_LAYER_INPUT)

// --- Kuantalama Ölçekleri (Scale Factors) ---
// Float özellikleri int16'ya çevirmek için giriş çarpanı
#define INPUT_SCALE          256.0f

// GUNCELLEME: degerler artik export/dataset__eval_heldout_test/
// model_weights_int16.h'dan (35 dosyayla egitilmis, 8 dosya gercekten
// hic gorulmemis, tez icin secilen model -- full.npz'nin DEGIL).
#define CONV1_WEIGHT_SCALE   85931.8558395006f
#define CONV1_BIAS_SCALE     214816.2626566571f

#define CONV2_WEIGHT_SCALE   138716.7796454836f
#define CONV2_BIAS_SCALE     356554.2466089580f

#define FC_WEIGHT_SCALE      281670.4694246628f
#define FC_BIAS_SCALE        704577.0132013098f

#endif /* DEFINITIONS_H_ */
