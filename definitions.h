/*
 * definitions.h
 *
 *  Created on: Feb 11, 2026
 *      Author: canbe
 */

#ifndef DEFINITIONS_H_
#define DEFINITIONS_H_



// main.c ve convolution.c içindeki eski isim uyumluluğu için
#define IMAGE_H              32   // 32
#define IMAGE_W              12   // 12
#define NUM_FILTERS          6  // 6

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

// Toplam gereken çalışma belleği (workspace) boyutu:
// 1800 (Conv1 Çıktısı) + 900 (Pool1 Çıktısı) + 624 (Conv2 Çıktısı) = 3324 eleman
#define MAX_ELEMENTS         (CONV1_SIZE + POOL1_SIZE + CONV2_SIZE)

// Dense katmana giren düzleştirilmiş veri boyutu
#define DENSE_LAYER_INPUT    288

#define DENSE_LAYER_WEIGHTS  (864) // 288 * 3 = 864

// --- Kuantalama Ölçekleri (Scale Factors) ---
// Float özellikleri int16'ya çevirmek için giriş çarpanı
#define INPUT_SCALE          256.0f

// Değerler model_weights_int16.h dosyasından alınmıştır
#define CONV1_WEIGHT_SCALE   85721.9989368545f
#define CONV1_BIAS_SCALE     88490.2848561893f

#define CONV2_WEIGHT_SCALE   179186.7648735111f
#define CONV2_BIAS_SCALE     226969.5285718945f

#define FC_WEIGHT_SCALE      300475.8649717340f
#define FC_BIAS_SCALE        897049.6296538947f

#endif /* DEFINITIONS_H_ */
