"""Scalar data types: how each Kernel IR dtype lives in PTX registers and memory.

Every value the compiler handles has one of these dtypes. A dtype answers three
questions for the code generator:

* which register class holds it (``%f`` for f32, ``%rd`` for i64, ...),
* which PTX type suffix loads and stores it (``ld.global.f32``),
* how big one element is in memory (for address arithmetic).

The typing rules follow emmy's own renderer so that our kernels compute what
emmy's CUDA would compute: f16 is a storage type whose arithmetic stays f16
only when every operand is f16, f64 values are computed in f32, and integer
arithmetic stays in the integer family.
"""

from __future__ import annotations

from dataclasses import dataclass

from emmyptx.errors import CompileError


@dataclass(frozen=True)
class DType:
    name: str  # the IR's canonical token: "f32", "f16", "i32", ...
    reg: str  # register class (see ptx.REG_CLASSES)
    mem: str  # PTX type suffix for ld / st / cvt
    size: int  # bytes per element
    kind: str  # "float", "int" (signed), "uint" or "bool"

    @property
    def is_float(self) -> bool:
        return self.kind == "float"

    @property
    def is_int(self) -> bool:
        return self.kind in ("int", "uint")


F16 = DType("f16", "b16", "f16", 2, "float")
F32 = DType("f32", "f32", "f32", 4, "float")
F64 = DType("f64", "f64", "f64", 8, "float")
I16 = DType("i16", "s32", "s16", 2, "int")
I32 = DType("i32", "s32", "s32", 4, "int")
I64 = DType("i64", "s64", "s64", 8, "int")
U8 = DType("u8", "s32", "u8", 1, "uint")
U16 = DType("u16", "s32", "u16", 2, "uint")
U32 = DType("u32", "s32", "u32", 4, "uint")
U64 = DType("u64", "s64", "u64", 8, "uint")
BOOL = DType("bool", "s32", "u8", 1, "bool")  # 0 or 1 in a register, one byte in memory

BY_NAME: dict[str, DType] = {d.name: d for d in (F16, F32, F64, I16, I32, I64, U8, U16, U32, U64, BOOL)}

# Shared-memory arrays name their element type in CUDA spelling.
_SMEM_SPELLING = {
    "float": F32,
    "__half": F16,
    "half": F16,
    "double": F64,
    "int": I32,
    "long long": I64,
    "unsigned int": U32,
    "unsigned char": U8,
    "bool": BOOL,
}


def get(dtype) -> DType:
    """Resolve an emmy ``DataType``, a canonical token, or ``None`` (meaning f32)."""
    if dtype is None:
        return F32
    if isinstance(dtype, DType):
        return dtype
    name = dtype if isinstance(dtype, str) else dtype.name
    if name in BY_NAME:
        return BY_NAME[name]
    if name in _SMEM_SPELLING:
        return _SMEM_SPELLING[name]
    raise CompileError(f"dtype {name!r} is not supported")


def compute_dtype(storage: DType) -> DType:
    """The dtype a freshly loaded value is computed in. f64 is storage only:
    emmy declares f64 elements as ``float`` locals, so we do the same."""
    return F32 if storage is F64 else storage


# Integer ops that stay in the integer family when all operands are integers.
INTEGER_OPS = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "floor_divide",
        "remainder",
        "mod",
        "left_shift",
        "right_shift",
        "bitwise_and",
        "bitwise_or",
        "bitwise_xor",
        "bitwise_count",
    }
)


def promote(op: str, args: list[DType]) -> DType:
    """Result dtype of an elementwise op, mirroring emmy's ``dtype_promote``:
    integers stay integers (C-style rank rule), all-f16 stays f16, anything
    else is f32. An f16 result is computed in f32 and rounded once."""
    if args and op in INTEGER_OPS and all(a.is_int for a in args):
        widest_signed = max((a.size for a in args if a.kind == "int"), default=0)
        widest_unsigned = max((a.size for a in args if a.kind == "uint"), default=0)
        if widest_unsigned >= widest_signed:
            return U64 if widest_unsigned == 8 else U32
        return I64 if widest_signed == 8 else I32
    if args and all(a is F16 for a in args):
        return F16
    return F32
