// Golden vector board test -- CCS Debug Server Scripting (DSS).
//
// For every <label> <out> pair: flash the .out over the XDS100v2, let the
// golden-vector loop in main.c run for RUN_MS, halt, and read back
//   g_total_count / g_correct_count   (every class must be predicted right)
//   g_last_logits[class][k]           (compared with EXPECTED below)
//
// usage (run_board_golden_test.sh calls this):
//   dss.bat golden_test.js <ccxml> <expected.txt> <label> <out> [<label> <out> ...]
//
// expected.txt: one line per class, "name l0 l1 l2", produced by the PC
// emulation of the very same sources (see README.md).

importPackage(Packages.com.ti.debug.engine.scripting);
importPackage(Packages.com.ti.ccstudio.scripting.environment);
importPackage(Packages.java.lang);
importPackage(Packages.java.io);

var RUN_MS = 10000;          // golden loop: ~dozens of full cycles in 10 s
var REL_TOL = 0.001;         // 0.1%: the all-CPU build matches the PC to 0.006%; a looser
                             // bound (1%) let a missing CLA-side fc_bias (0.7%) slip through

function readExpected(path) {
    var rows = [], names = [];
    var br = new BufferedReader(new FileReader(path));
    var line;
    while ((line = br.readLine()) != null) {
        line = String(line).replace(/^\s+|\s+$/g, "");
        if (line.length == 0 || line.charAt(0) == "#") continue;
        var p = line.split(/\s+/);
        names.push(p[0]);
        rows.push([parseFloat(p[1]), parseFloat(p[2]), parseFloat(p[3])]);
    }
    br.close();
    return { names: names, rows: rows };
}

function toFloat32(bits) {
    // DSS hands back an unsigned 32-bit value; turn it into a signed int for
    // Float.intBitsToFloat
    if (bits > 2147483647) bits -= 4294967296;
    // Rhino already converts the returned Java float to a JS number
    return Number(Float.intBitsToFloat(new Integer(bits)));
}

var ccxml = arguments[0];
var exp = readExpected(arguments[1]);
var NC = exp.rows.length;

// --cold-boot: after flashing, wipe the CLA data RAM (RAMLS0-4) and start from
// the flash entry point 0x80000 -- the address the boot ROM jumps to on a real
// power-up. The debugger normally writes initialized RAM sections while it
// loads the program; this mode checks the firmware still works when nothing
// but flash survives (i.e. without a debugger attached).
var COLD = false;
var firstPair = 2;
if (arguments[2] == "--cold-boot") { COLD = true; firstPair = 3; }
var LS_START = 0x8000, LS_WORDS = 0x2800, FLASH_ENTRY = 0x80000;

function wipeClaDataRam() {
    var CHUNK = 256;
    var zeros = java.lang.reflect.Array.newInstance(java.lang.Long.TYPE, CHUNK);
    for (var off = 0; off < LS_WORDS; off += CHUNK)
        // explicit overload: Rhino cannot pick between (…,long[],int) and (…,long,int)
        session.memory["writeData(int,long,long[],int)"](Memory.Page.DATA, LS_START + off, zeros, 16);
}

var env = ScriptingEnvironment.instance();
env.traceSetConsoleLevel(TraceLevel.OFF);
var server = env.getServer("DebugServer.1");
server.setConfig(ccxml);

var session;
try {
    session = server.openSession(".*C28xx_CPU1");
} catch (e) {
    print("ERROR: could not open a C28xx_CPU1 session -- " + e);
    java.lang.System.exit(2);
}
try {
    session.target.connect();
} catch (e) {
    print("ERROR: could not connect to the board (is the LaunchXL plugged in and " +
          "the XDS100v2 driver installed?) -- " + e);
    server.stop();
    java.lang.System.exit(2);
}

var summary = [];
for (var a = firstPair; a + 1 < arguments.length; a += 2) {
    var label = arguments[a], out = arguments[a + 1];
    var ok = true, notes = [];
    print("\n=== " + label + (COLD ? "  [cold boot]" : "") + "   " + out);
    try {
        session.target.reset();
        print("  flashing...");
        session.memory.loadProgram(out);
        if (COLD) {
            wipeClaDataRam();
            var probe = session.symbol.getAddress("weight");
            var w0 = session.memory.readData(Memory.Page.DATA, probe, 16, 1)[0];
            print("  CLA data RAM wiped (weight[0] now reads " + w0 + "), starting at 0x80000");
            session.memory.writeRegister("PC", FLASH_ENTRY);
        }
        session.target.runAsynch();
        Thread.sleep(RUN_MS);
        session.target.halt();

        var total = session.expression.evaluate("g_total_count");
        var correct = session.expression.evaluate("g_correct_count");
        print("  golden loop: " + correct + "/" + total + " predictions correct");
        if (total < NC) { ok = false; notes.push("loop made too little progress (" + total + " iterations)"); }
        if (correct != total) { ok = false; notes.push((total - correct) + " wrong predictions"); }

        var addr = session.symbol.getAddress("g_last_logits");
        var raw = session.memory.readData(Memory.Page.DATA, addr, 32, NC * NC);
        var worst = 0;
        for (var c = 0; c < NC; c++) {
            var got = [], line = "  " + exp.names[c] + ":";
            for (var k = 0; k < NC; k++) {
                var v = toFloat32(raw[c * NC + k]);
                got.push(v);
                var ref = exp.rows[c][k];
                var rel = Math.abs(v - ref) / Math.max(1.0, Math.abs(ref));
                if (rel > worst) worst = rel;
                line += "  " + v.toFixed(3) + " (exp " + ref.toFixed(3) + ")";
            }
            var arg = 0;
            for (var k2 = 1; k2 < NC; k2++) if (got[k2] > got[arg]) arg = k2;
            if (arg != c) { ok = false; notes.push(exp.names[c] + " predicted as " + exp.names[arg]); }
            print(line);
        }
        print("  worst logit deviation from PC emulation: " + (100 * worst).toFixed(3) + "%");
        if (worst > REL_TOL) { ok = false; notes.push("logits differ from PC emulation by > " + (100 * REL_TOL) + "%"); }
    } catch (e) {
        ok = false; notes.push("exception: " + e);
    }
    print("  RESULT " + label + ": " + (ok ? "PASS" : "FAIL  -- " + notes.join("; ")));
    summary.push(label + ": " + (ok ? "PASS" : "FAIL"));
}

session.target.disconnect();
server.stop();
print("\n==== SUMMARY ====\n  " + summary.join("\n  "));
java.lang.System.exit(summary.join(" ").indexOf("FAIL") >= 0 ? 1 : 0);
