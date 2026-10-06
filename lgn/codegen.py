"""Emit a netlist as a standalone C++ function.

The generated `lgn_eval` takes 64 samples at once, bit-sliced: x[i] holds input
bit i for all 64 samples. Each gate is a single 64-bit bitwise instruction, so
there are no multiplies, no floats and no memory traffic beyond the inputs.
"""

from .circuit import CONST0, CONST1, Netlist

_EXPR = {
    0: "0", 1: "{a} & {b}", 2: "{a} & ~{b}", 3: "{a}", 4: "~{a} & {b}", 5: "{b}",
    6: "{a} ^ {b}", 7: "{a} | {b}", 8: "~({a} | {b})", 9: "~({a} ^ {b})", 10: "~{b}",
    11: "{a} | ~{b}", 12: "~{a}", 13: "~{a} | {b}", 14: "~({a} & {b})", 15: "~(uint64_t)0",
}


def _name(net: Netlist, r: int) -> str:
    if r == CONST0:
        return "(uint64_t)0"
    if r == CONST1:
        return "~(uint64_t)0"
    return f"v[{r}]"


def to_cpp(net: Netlist, chunk: int = 1500) -> str:
    """Straight-line C++, split into small functions.

    One giant function with tens of thousands of statements takes the compiler
    minutes to optimise, so gates are written into a shared array `v` (inputs
    first, then gates) in chunks of `chunk` statements.
    """
    n_in, n_nodes = net.n_inputs, net.n_inputs + net.n_gates
    lines = [
        "#include <cstdint>",
        "#include <cstring>",
        "",
        f"constexpr int LGN_INPUTS = {n_in};",
        f"constexpr int LGN_OUTPUTS = {len(net.outputs)};",
        f"constexpr int LGN_GATES = {net.n_gates};",
        "",
        "namespace {",
        "",
    ]
    n_chunks = 0
    for start in range(0, net.n_gates, chunk):
        lines.append(f"void lgn_chunk{n_chunks}(uint64_t *__restrict v) {{")
        for j in range(start, min(start + chunk, net.n_gates)):
            op, a, b = int(net.ops[j]), int(net.a[j]), int(net.b[j])
            expr = _EXPR[op].format(a=_name(net, a), b=_name(net, b))
            lines.append(f"    v[{n_in + j}] = {expr};")
        lines.append("}")
        n_chunks += 1
    lines += ["", "}  // namespace", "",
              "void lgn_eval(const uint64_t *__restrict x, uint64_t *__restrict y) {",
              f"    static uint64_t v[{max(n_nodes, 1)}];",
              "    std::memcpy(v, x, sizeof(uint64_t) * LGN_INPUTS);"]
    lines += [f"    lgn_chunk{i}(v);" for i in range(n_chunks)]
    lines += [f"    y[{k}] = {_name(net, int(r))};" for k, r in enumerate(net.outputs)]
    lines.append("}")
    return "\n".join(lines) + "\n"


BENCH_MAIN = r"""
#include <bit>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

void lgn_eval(const uint64_t *__restrict x, uint64_t *__restrict y);

static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

// usage: bench packed_inputs.bin n_words n_classes out_counts.bin
int main(int argc, char **argv) {
    if (argc != 5) { std::fprintf(stderr, "usage: %s in.bin n_words n_classes out.bin\n", argv[0]); return 1; }
    int W = std::atoi(argv[2]), C = std::atoi(argv[3]), per = LGN_OUTPUTS / C;
    std::vector<uint64_t> x((size_t)LGN_INPUTS * W);  // word-major: x[w*IN + i]
    uint64_t y[LGN_OUTPUTS];
    {
        std::ifstream f(argv[1], std::ios::binary);
        if (!f.read(reinterpret_cast<char *>(x.data()), x.size() * sizeof(uint64_t))) return 1;
    }

    // circuit only
    volatile uint64_t sink = 0;
    int reps = 1; double t0, dt;
    do {
        reps *= 2; t0 = now();
        for (int r = 0; r < reps; r++)
            for (int w = 0; w < W; w++) { lgn_eval(&x[(size_t)w * LGN_INPUTS], y); sink = sink ^ y[0]; }
        dt = now() - t0;
    } while (dt < 1.0);
    double ns_circuit = dt * 1e9 / ((double)reps * W * 64);

    // end to end: circuit + vote counting, written out for checking
    std::vector<int32_t> counts((size_t)W * 64 * C, 0);
    t0 = now();
    for (int w = 0; w < W; w++) {
        lgn_eval(&x[(size_t)w * LGN_INPUTS], y);
        for (int k = 0; k < LGN_OUTPUTS; k++) {
            uint64_t bits = y[k]; int c = k / per;
            while (bits) { int lane = std::countr_zero(bits); counts[((size_t)w * 64 + lane) * C + c]++; bits &= bits - 1; }
        }
    }
    double ns_total = (now() - t0) * 1e9 / ((double)W * 64);

    std::ofstream(argv[4], std::ios::binary).write(reinterpret_cast<const char *>(counts.data()), counts.size() * sizeof(int32_t));
    std::printf("{\"ns_per_sample_circuit\": %.3f, \"ns_per_sample_with_votes\": %.3f}\n", ns_circuit, ns_total);
    return 0;
}
"""
