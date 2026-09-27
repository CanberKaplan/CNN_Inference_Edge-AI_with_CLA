#!/usr/bin/env bash
# Builds the firmware for the requested CPU/CLA stage configurations and runs
# the golden-vector test on a LaunchXL-F28379D over its XDS100v2.
#
#   ./run_board_golden_test.sh                  # configs 111 101 000
#   ./run_board_golden_test.sh 111 110 001      # any subset of S1S2S3
#   ./run_board_golden_test.sh --build-only     # just compile/link, no board
#
# A config is three digits S1 S2 S3 (1 = CLA, 0 = CPU), matching
# cla_pipeline_config.h. Each config is built from a copy of the sources in
# $BUILD_DIR/<cfg>/ -- by default OUTSIDE the project, in the system temp dir.
# It must stay outside: CCS pulls every source and .cmd file under the project
# folder into its own build, so copies inside the project made CCS link 17
# linker command files at once (#10263 "memory range has already been
# specified" x83). The project's own cla_pipeline_config.h and Debug/ are never
# touched. Uses the same compiler/linker flags as the CCS project.
set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
FW="$(cd "$HERE/../.." && pwd)"
CCS="${CCS:-C:/ti/ccs1271/ccs}"
CGT="${CGT:-$CCS/tools/compiler/ti-cgt-c2000_22.6.1.LTS}"
C2KW="${C2KW:-C:/ti/c2000/C2000Ware_5_05_00_00}"
CL="$CGT/bin/cl2000"
DSS="$CCS/ccs_base/scripting/bin/dss.bat"
CCXML="$FW/targetConfigs/TMS320F28379D.ccxml"

if [ -n "${BUILD_DIR:-}" ]; then
  BUILD_ROOT="$BUILD_DIR"
else
  tmp="${TEMP:-/tmp}"
  command -v cygpath >/dev/null 2>&1 && tmp="$(cygpath -u "$tmp")"
  BUILD_ROOT="$tmp/cnn_cla_board_golden_test"
fi
mkdir -p "$BUILD_ROOT"
BUILD_ROOT="$(cd "$BUILD_ROOT" && pwd)"
case "$BUILD_ROOT/" in
  "$FW"/*) echo "BUILD_DIR must be outside the CCS project ($FW): CCS would compile the copies"; exit 2 ;;
esac

build_only=0; cold=(); cfgs=()
for a in "$@"; do
  case "$a" in
    --build-only) build_only=1 ;;
    --cold-boot)  cold=(--cold-boot) ;;
    [01][01][01]) cfgs+=("$a") ;;
    *) echo "unknown argument: $a"; exit 2 ;;
  esac
done
[ ${#cfgs[@]} -eq 0 ] && cfgs=(111 101 000)

[ -x "$CL" ] || { echo "cl2000 not found at $CL (set CGT=...)"; exit 2; }

FLAGS="-v28 -ml -mt --cla_support=cla1 --float_support=fpu32 --tmu_support=tmu0 --vcu_support=vcu2 --fp_mode=relaxed --define=_LAUNCHXL_F28379D --define=_FLASH --define=CPU1 -g --diag_warning=225 --diag_wrap=off --display_error_number --abi=coffabi --cla_signed_compare_workaround=on"
SRCS="F2837xD_Adc.c F2837xD_CodeStartBranch.asm F2837xD_DefaultISR.c F2837xD_Dma.c F2837xD_GlobalVariableDefs.c F2837xD_Gpio.c F2837xD_Ipc.c F2837xD_PieCtrl.c F2837xD_PieVect.c F2837xD_SysCtrl.c F2837xD_usDelay.asm convolution.c dense_layer.cla main.c mfcc_extract.c firmware/source/adc.c"

winpath() { if command -v cygpath >/dev/null 2>&1; then cygpath -w "$1"; else echo "$1"; fi; }

build_cfg() {  # $1 = S1S2S3
  local cfg="$1" B="$BUILD_ROOT/$1"
  local s1="${cfg:0:1}" s2="${cfg:1:1}" s3="${cfg:2:1}"
  rm -rf "$B"; mkdir -p "$B/src" "$B/obj"
  ( cd "$FW" && tar -c --exclude=.git --exclude=Debug --exclude=tools --exclude=.metadata --exclude=.jxbrowser.userdata . ) | tar -x -C "$B/src"
  sed -i -e "s/^#define CLA_STAGE_CONV1_POOL1 .*/#define CLA_STAGE_CONV1_POOL1  $s1/" \
         -e "s/^#define CLA_STAGE_CONV2_POOL2 .*/#define CLA_STAGE_CONV2_POOL2  $s2/" \
         -e "s/^#define CLA_STAGE_DENSE .*/#define CLA_STAGE_DENSE        $s3/" "$B/src/cla_pipeline_config.h"
  local objs=""
  for s in $SRCS; do
    ( cd "$B/src" && "$CL" $FLAGS \
        --include_path="$B/src/firmware/include" \
        --include_path="$C2KW/device_support/f2837xd/common/include" \
        --include_path="$C2KW/device_support/f2837xd/headers/include" \
        --include_path="$B/src" --include_path="$CGT/include" \
        --obj_directory="$B/obj" "$s" ) >>"$B/build.log" 2>&1 \
      || { echo "  [$cfg] COMPILE FAILED on $s -- see $B/build.log"; return 1; }
    b="$(basename "$s")"; objs="$objs $B/obj/${b%.*}.obj"
  done
  ( cd "$B" && "$CL" $FLAGS -z -m"$B/fw.map" --stack_size=0x200 --warn_sections \
      -i"$CGT/lib" -i"$CGT/include" --reread_libs --define=CLA_C --rom_model \
      -o "$B/fw.out" $objs "$B/src/2837xD_FLASH_CLA_lnk_cpu1.cmd" \
      -l"libc.a" -l"$C2KW/device_support/f2837xd/headers/cmd/F2837xD_Headers_nonBIOS_cpu1.cmd" \
  ) >>"$B/build.log" 2>&1 || { echo "  [$cfg] LINK FAILED -- see $B/build.log"; return 1; }
  local prog data
  prog=$(awk '$1=="RAMLS5_PROG"{printf "%d/%d", strtonum("0x"$4), strtonum("0x"$3)}' "$B/fw.map")
  data=$(awk '$1=="RAMLS_0_1_2_3_4"{printf "%d/%d", strtonum("0x"$4), strtonum("0x"$3)}' "$B/fw.map")
  echo "  [$cfg] built   CLA program $prog words   CLA data $data words"
}

echo "building configs: ${cfgs[*]}"
dss_args=(); failed=0
for c in "${cfgs[@]}"; do
  if build_cfg "$c"; then
    dss_args+=("cfg_$c" "$(winpath "$BUILD_ROOT/$c/fw.out")")
  else
    failed=1
  fi
done
[ $failed -eq 0 ] || { echo "build failed, not touching the board"; exit 1; }
[ $build_only -eq 1 ] && { echo "build-only: done"; exit 0; }

[ -f "$DSS" ] || { echo "DSS not found at $DSS (set CCS=...)"; exit 2; }
echo; echo "running golden-vector test on the board..."
"$DSS" "$(winpath "$HERE/golden_test.js")" "$(winpath "$CCXML")" \
       "$(winpath "$HERE/expected_logits.txt")" "${cold[@]}" "${dss_args[@]}"
