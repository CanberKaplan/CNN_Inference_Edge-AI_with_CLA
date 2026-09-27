# Golden vector board test

Builds the firmware for chosen CPU/CLA stage splits and checks, on a
LaunchXL-F28379D, that the golden-vector loop in `main.c` classifies all three
classes correctly and produces the same logits as the PC emulation.

## Run

Plug in the LaunchXL (onboard XDS100v2) and **close any CCS debug session**
(CCS and DSS cannot share the probe). From Git Bash:

```bash
./run_board_golden_test.sh                 # configs 111 101 000
./run_board_golden_test.sh 110 011 001     # any S1S2S3 (1 = CLA, 0 = CPU)
./run_board_golden_test.sh --build-only    # compile/link only, no board
./run_board_golden_test.sh --cold-boot     # simulate a power-up without the debugger
```

`--cold-boot` flashes the program, zeroes the CLA data RAM (RAMLS0-4), and
starts at the flash entry point 0x80000, where the boot ROM jumps on a real
power-up. Normally the debugger writes initialized RAM sections while it loads
the program, and that can hide firmware that would not work stand-alone. This
mode caught exactly that bug: the CLA weights in `CLADataLS1` had no flash
copy. Config 000 keeps all weights in flash, so it is a good control run.

Each config is built from a copy of the sources in the system temp dir
(`%TEMP%/cnn_cla_board_golden_test/<cfg>/`, override with `BUILD_DIR=`), with that
config written into the copy's `cla_pipeline_config.h`. The project's own
`cla_pipeline_config.h` and `Debug/` are left alone. Compiler/linker flags are
the CCS project's (TI C2000 CGT 22.6.1). Override tool locations with `CCS=`,
`CGT=`, `C2KW=` if yours differ.

## What PASS means (per config)

After flashing, the loop runs for 10 s and is halted. Then:

- `g_correct_count == g_total_count` (every iteration predicted the right class)
- argmax of `g_last_logits[c]` is `c` for each class
- every logit is within 0.1 % of `expected_logits.txt` (the all-CPU build
  matches the PC emulation to ~0.006 %; a 1 % bound once hid a real CLA bug)

## expected_logits.txt

These values come from running the same sources on the PC: `dense_layer.cla` and
`convolution.c` are compiled with gcc and the TI-only keywords are stubbed out.
All 8 stage splits and the single-task `Cla1Task1` gave bit-identical results.
Regenerate the file whenever the weights, `INPUT_SCALE` or the kernels change,
otherwise the logit check will fail even on correct firmware.

## Keep build output outside the CCS project

CCS adds every source and `.cmd` file under the project folder to its own
build, including files in subfolders. The tool used to keep its per-config
source copies in `tools/board_golden_test/build/`. CCS then linked 17 linker
command files at once and failed with `#10263 ... memory range has already been
specified` (83 errors). The copies now go to the temp dir, and the script
refuses a `BUILD_DIR` inside the project. As an extra safeguard, right-click
`tools` in CCS → *Exclude from Build*.
