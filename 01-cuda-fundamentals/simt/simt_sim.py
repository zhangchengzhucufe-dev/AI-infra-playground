"""SIMT simulator: the execution model of one warp, on CPU.

No GPU required.

Contract: run(program) -> (regs, cycles)
- a warp is fixed at 32 lanes; lane i starts with reg = i (int)
- program is a list of instructions; each instruction is a tuple, one of:
    ("add", k)   reg += k on active lanes, 1 cycle
    ("mul", k)   reg *= k on active lanes, 1 cycle
    ("if_lt", t, then_prog, else_prog)
        lanes with reg < t execute then_prog, the rest execute else_prog.
        Run then_prog under its mask, then else_prog under the
        complementary mask, then reconverge. A branch with no active
        lanes is skipped entirely and costs no cycles. Nested
        instructions still count cycles (that is where divergence costs).
        The if_lt itself is free; cycles come only from the add / mul
        actually executed.
- regs: final register values of the 32 lanes (list); cycles: total count.

Done when `pytest tests/test_simt_sim.py` passes.
"""


def run(program):
    regs = list(range(32))

    # exec_block returns the cycles consumed by this block; only active lanes are touched.
    def exec_block(prog, active):
        cycles = 0
        for inst in prog:
            if inst[0] == "add":
                k = inst[1]
                for i in range(32):
                    if active[i]:
                        regs[i] += k
                cycles += 1
            elif inst[0] == "mul":
                k = inst[1]
                for i in range(32):
                    if active[i]:
                        regs[i] *= k
                cycles += 1
            elif inst[0] == "if_lt":
                t, then_prog, else_prog = inst[1], inst[2], inst[3]
                # split the active lanes by reg < t; each branch runs on
                # "parent mask AND branch condition".
                then_active = [active[i] and regs[i] < t for i in range(32)]
                else_active = [active[i] and regs[i] >= t for i in range(32)]
                if any(then_active):
                    cycles += exec_block(then_prog, then_active)
                if any(else_active):
                    cycles += exec_block(else_prog, else_active)
                # reconverge: if_lt itself costs no cycles.
            else:
                raise ValueError(f"unknown instruction: {inst!r}")
        return cycles

    cycles = exec_block(program, [True] * 32)
    return regs, cycles
