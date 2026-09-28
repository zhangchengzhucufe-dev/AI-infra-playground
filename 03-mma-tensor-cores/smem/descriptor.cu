// SM100's 64-bit smem matrix descriptor: the field packing, plus the
// LBO/SBO/layout values for three smem layouts of a 64x64 bf16 B tile
// (to be consumed by tcgen05):
//   Scenario 1: K-major, no swizzle, tile start smem address 0x1000.
//           The canonical layout's atom is 8 rows x 16B tightly packed
//           (128B contiguous); LBO/SBO follow from the atom strides
//           along K and along MN.
//   Scenario 2: K-major, 128B swizzle, start address 0x2000.
//   Scenario 3: MN-major, 128B swizzle, start address 0x3000.
//
// make_desc packs the parameters into the 64 bits field by field.
// SM100 fields (note the differences from the sm90 wgmma version):
// start_address bits[0,14) = addr >> 4;
// LBO bits[16,30) = lbo >> 4; SBO bits[32,46) = sbo >> 4;
// version bits[46,48) fixed at 1; layout_type bits[61,64), 3 bits:
// NONE=0, 128B=2, 64B=4, 32B=6 (sm90 uses 2 bits and 128B=1).
//
// With swizzle, LBO is ignored by hardware -- filled 0. Scenarios 2 and
// 3 come out identical: at 64x64 bf16 the SBO is 1024B in either
// direction, and the K-major vs MN-major difference would only show up
// in LBO, which the swizzled layouts ignore.
//
// The expected values were exercised with tcgen05 on a B300 (borrowed
// time on a lab machine), so they reflect what real hardware reads.
// The check itself is host-only and runs anywhere.
// Run: make run/smem/descriptor
#include <cstdio>
#include <cstdint>

// Field encoding: each value is shifted right by 4 first (16B
//     granularity), then placed into its own bit field; version is
//     fixed at 1, and layout takes bits[61,64) directly.
static uint64_t make_desc(uint32_t saddr, uint32_t lbo, uint32_t sbo,
                          uint32_t layout) {
    uint64_t d = 0;
    d |= (uint64_t)((saddr >> 4) & 0x3FFFu);
    d |= (uint64_t)((lbo >> 4) & 0x3FFFu) << 16;
    d |= (uint64_t)((sbo >> 4) & 0x3FFFu) << 32;
    d |= (uint64_t)1 << 46;  // version = 1
    d |= (uint64_t)(layout & 0x7u) << 61;
    return d;
}

// Scenario 1 (K-major, no swizzle): atom = 8k x 8n (16B), the 8 n
//     within an atom pack tightly into 128B. Leading direction is
//     along K: adjacent atoms (K+8) are 128B apart -> LBO=128; strided
//     direction is along N: adjacent atoms (N+8) skip a whole n group
//     = 8n x 64k x 2B = 1024B -> SBO=1024; layout = NONE.
// Scenario 2 (K-major, 128B swizzle): LBO is ignored by hardware, fill
//     0; after swizzling, atoms still move in 128B blocks and n groups
//     are still 1024B apart -> SBO=1024; layout = 128B (encoding 2).
// Scenario 3 (MN-major, 128B swizzle): atom = 8n x 16B contiguous
//     along N; strided direction along K: 8k x 64n x 2B = 1024B ->
//     SBO=1024; LBO ignored, fill 0; layout = 128B.
//     (The 64x64 tile makes SBO come out the same in both directions,
//     so this is exactly identical to scenario 2.)
static const uint32_t SCEN[3][3] = {
    {128, 1024, 0},  // scenario 1: K-major, no swizzle
    {0, 1024, 2},    // scenario 2: K-major, 128B swizzle
    {0, 1024, 2},    // scenario 3: MN-major, 128B swizzle
};

// Check below. On mismatch it reports the differing
// fields, without printing the expected values.
static const uint32_t ADDR[3] = {0x1000, 0x2000, 0x3000};
static const uint64_t TRUTH[3] = {0x0000404000080100ull, 0x4000404000000200ull,
                                  0x4000404000000300ull};

static void field_diff(uint64_t got, uint64_t want) {
    struct { const char* name; int lo, w; } f[] = {
        {"start_address", 0, 14}, {"LBO", 16, 14}, {"SBO", 32, 14},
        {"version", 46, 2}, {"base_offset", 49, 3}, {"layout_type", 61, 3}};
    for (auto& x : f) {
        uint64_t g = (got >> x.lo) & ((1ull << x.w) - 1);
        uint64_t w = (want >> x.lo) & ((1ull << x.w) - 1);
        if (g != w) printf("    field %s mismatch (yours=0x%llx)\n", x.name,
                           (unsigned long long)g);
    }
}

int main() {
    int bad = 0;
    for (int i = 0; i < 3; i++) {
        uint64_t got = make_desc(ADDR[i], SCEN[i][0], SCEN[i][1], SCEN[i][2]);
        if (got != TRUTH[i]) {
            printf("scenario %d FAIL:\n", i + 1);
            field_diff(got, TRUTH[i]);
            bad++;
        } else {
            printf("scenario %d PASS\n", i + 1);
        }
    }
    return bad != 0;
}
