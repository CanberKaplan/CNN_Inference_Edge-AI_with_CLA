################################################################################
# Automatically-generated file. Do not edit!
################################################################################

SHELL = cmd.exe

# Add inputs and outputs from these tool invocations to the build variables 
CMD_SRCS += \
../2837xD_FLASH_CLA_lnk_cpu1.cmd 

CLA_SRCS += \
../dense_layer.cla 

ASM_SRCS += \
../F2837xD_CodeStartBranch.asm \
../F2837xD_usDelay.asm 

C_SRCS += \
../F2837xD_DefaultISR.c \
../F2837xD_GlobalVariableDefs.c \
../F2837xD_Gpio.c \
../F2837xD_PieCtrl.c \
../F2837xD_PieVect.c \
../F2837xD_SysCtrl.c \
../convolution.c \
../main.c 

CLA_DEPS += \
./dense_layer.d 

C_DEPS += \
./F2837xD_DefaultISR.d \
./F2837xD_GlobalVariableDefs.d \
./F2837xD_Gpio.d \
./F2837xD_PieCtrl.d \
./F2837xD_PieVect.d \
./F2837xD_SysCtrl.d \
./convolution.d \
./main.d 

OBJS += \
./F2837xD_CodeStartBranch.obj \
./F2837xD_DefaultISR.obj \
./F2837xD_GlobalVariableDefs.obj \
./F2837xD_Gpio.obj \
./F2837xD_PieCtrl.obj \
./F2837xD_PieVect.obj \
./F2837xD_SysCtrl.obj \
./F2837xD_usDelay.obj \
./convolution.obj \
./dense_layer.obj \
./main.obj 

ASM_DEPS += \
./F2837xD_CodeStartBranch.d \
./F2837xD_usDelay.d 

OBJS__QUOTED += \
"F2837xD_CodeStartBranch.obj" \
"F2837xD_DefaultISR.obj" \
"F2837xD_GlobalVariableDefs.obj" \
"F2837xD_Gpio.obj" \
"F2837xD_PieCtrl.obj" \
"F2837xD_PieVect.obj" \
"F2837xD_SysCtrl.obj" \
"F2837xD_usDelay.obj" \
"convolution.obj" \
"dense_layer.obj" \
"main.obj" 

C_DEPS__QUOTED += \
"F2837xD_DefaultISR.d" \
"F2837xD_GlobalVariableDefs.d" \
"F2837xD_Gpio.d" \
"F2837xD_PieCtrl.d" \
"F2837xD_PieVect.d" \
"F2837xD_SysCtrl.d" \
"convolution.d" \
"main.d" 

CLA_DEPS__QUOTED += \
"dense_layer.d" 

ASM_DEPS__QUOTED += \
"F2837xD_CodeStartBranch.d" \
"F2837xD_usDelay.d" 

ASM_SRCS__QUOTED += \
"../F2837xD_CodeStartBranch.asm" \
"../F2837xD_usDelay.asm" 

C_SRCS__QUOTED += \
"../F2837xD_DefaultISR.c" \
"../F2837xD_GlobalVariableDefs.c" \
"../F2837xD_Gpio.c" \
"../F2837xD_PieCtrl.c" \
"../F2837xD_PieVect.c" \
"../F2837xD_SysCtrl.c" \
"../convolution.c" \
"../main.c" 


