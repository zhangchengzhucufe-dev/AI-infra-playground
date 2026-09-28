// Address mapping of the three smem swizzle modes.
//
// Input: an element's logical coordinates (row within the atom, byte
// offset within the row); output: the physical byte offset within the
// atom. Atom sizes per mode:
//   128B swizzle: 8 rows x 128B; 64B: 8 rows (4 rows per period) x 64B;
//   32B: 8 rows (2 rows per period) x 32B.
// All three are the same address-bit XOR, per the swizzling section of
// the PTX ISA: the 16B-chunk index XORs with the low bits of the row
// within one period. Low bits below 16B always pass through unchanged.
//
// Two host-side checks (no GPU):
//   1) Bijection: the outputs of all (row, colByte) in an atom must
//      tile the atom exactly, with no collisions;
//   2) Conflict-free column access: fix a logical 16B chunk index j,
//      sweep the rows within one period, and the physical chunks must
//      all differ (this is what the descriptor path needs to read by
//      column; plain padding or an identity mapping fails right here).
// Run: make run/smem/swizzle
//
// swizzle_128B is the same layout TMA's SWIZZLE_128B mode produces when
// landing tiles in smem, so anything built against the descriptor.cu
// scenarios agrees with hardware staging: wrong layout means wrong GEMM.
#include <cstdio>
#include <cstring>

// General pattern: the row's 16B chunk index is XORed with low bits of
// the row number; how many bits take part depends on the row width --
// a 128B row has 8 chunks (3 bits) <-> low 3 bits of row; a 64B row
// has 4 chunks (2 bits) <-> low 2 bits; a 32B row has 2 chunks (1 bit)
// <-> low 1 bit. Bits below 16B are untouched. This is exactly what
// guarantees: fix a logical chunk, sweep the rows of one period, and
// the physical chunks all differ (conflict-free column access).
static int swizzle_128B(int row, int colByte) {
    return row * 128 + (colByte ^ ((row & 7) << 4));
}
static int swizzle_64B(int row, int colByte) {
    return row * 64 + (colByte ^ ((row & 3) << 4));
}
static int swizzle_32B(int row, int colByte) {
    return row * 32 + (colByte ^ ((row & 1) << 4));
}

// Checks below.
static int check_mode(const char* name, int (*fn)(int, int), int rowBytes,
                      int period) {
    const int rows = 8;
    int atom = rows * rowBytes;
    static char hit[8 * 128];
    memset(hit, 0, atom);
    int bad = 0;
    // Convention: the output is an offset within the atom; the row base
    // row*rowBytes is included by the mapping itself.
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < rowBytes; c++) {
            int off = fn(r, c);
            if (off < 0 || off >= atom || hit[off]) bad++;
            else hit[off] = 1;
        }
    if (bad) {
        printf("%s FAIL: bijection check, %d out-of-range or collided entries\n",
               name, bad);
        return 1;
    }
    // Column access: fix 16B chunk j; the rows within one period must
    // map to distinct physical chunks
    int chunks = rowBytes / 16;
    for (int j = 0; j < chunks; j++) {
        unsigned seen = 0;
        for (int r = 0; r < period; r++) {
            int off = fn(r, j * 16);
            int physChunk = (off % rowBytes) / 16;
            if (seen & (1u << physChunk)) {
                printf("%s FAIL: column %d repeats a physical chunk across rows 0..%d\n",
                       name, j, period - 1);
                return 1;
            }
            seen |= 1u << physChunk;
        }
    }
    printf("%s PASS\n", name);
    return 0;
}

int main() {
    int bad = 0;
    bad += check_mode("128B", swizzle_128B, 128, 8);
    bad += check_mode(" 64B", swizzle_64B, 64, 4);
    bad += check_mode(" 32B", swizzle_32B, 32, 2);
    return bad != 0;
}
