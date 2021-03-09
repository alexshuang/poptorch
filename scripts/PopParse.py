#!/usr/bin/env python3
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

import enum
import argparse
import logging
import os
import re
import sys

import clang.cindex
from popgen import onnx
from utils import _utils

logger = logging.getLogger("PopParse")
_utils.set_logger(logger)

parser = argparse.ArgumentParser()
parser.add_argument("-c",
                    "--clang",
                    type=str,
                    help="Manually set path to clang headers")
parser.add_argument("-D",
                    "--debug",
                    action='store_true',
                    help="Enable debug printing")

args = parser.parse_args()

popart_dir = onnx.find_popart_includes()
onnx.init(popart_dir, args.clang, args.debug)
jsonOutput = onnx.parse()

logging_level = logging.DEBUG if args.debug else logging.INFO
logging.basicConfig(level=logging_level)

# List of SessionOptions attributes PopTorch decided to not support
options_not_handled = [
    "enableLoadAndOffloadRNGState",
    "prefetchBufferingDepthMap",
    "accumulateOuterFragmentSettings",
    "tensorLocationSettingsOverride",
    # Handled by PopTorch but not detected by this parser:
    "activationTensorLocationSettings",
    "weightTensorLocationSettings",
    "optimizerStateTensorLocationSettings",
    "accumulatorTensorLocationSettings",
    "replicatedGraphCount",
    "accumulationReductionType",
    "executionPhaseSettings",
    "batchSerializationSettings",
]


class OptionType(enum.IntEnum):
    Bool = 0
    Int = 1
    Float = 2
    String = 3
    Container = 4
    Enum = 5
    Object = 6


# check container_options
def parse_session_options(root_node):  # pylint: disable=too-many-statements
    # Build the list of options handled by Poptorch:
    handled = {}
    checks = {
        r".*container_options, \"(.*)\",.*": OptionType.Container,
        r" *ADD_POPART_ENUM_OPTION\(([^,]+),.*": OptionType.Enum,
        r" *ADD_POPART_STRING_OPTION\((.*)\).*": OptionType.String,
        r" *ADD_POPART_UINT64_OPTION\((.*)\).*": OptionType.Int,
        r" *ADD_POPART_BOOL_OPTION\((.*)\).*": OptionType.Bool,
        r" *ADD_POPART_DOUBLE_OPTION\((.*)\).*": OptionType.Float
    }

    for line in open(
            os.path.join(_utils.sources_dir(), "popart_compiler", "source",
                         "SessionOptions.cpp"), "r"):
        for expr, type in checks.items():
            m = re.match(expr, line)
            if m:
                handled[m.group(1)] = type
                break

    def find_session_options(node):
        if node.kind == clang.cindex.CursorKind.STRUCT_DECL and \
                node.spelling == "SessionOptions":
            return node

        for c in node.get_children():
            n = find_session_options(c)
            if n:
                return n
        return None

    def get_child(parent, child_type):
        child = None
        for c in parent.get_children():
            if c.kind == child_type:
                assert child is None, (
                    f"More than one child of "
                    f"{parent.spelling} has type {str(child_type)}")
                child = c
        return child

    opts = find_session_options(root_node)
    expected = {}
    # Build the list of attributes in Popart's SessionOptions
    for c in opts.get_children():
        if c.kind != clang.cindex.CursorKind.FIELD_DECL:
            continue
        children = list(c.get_children())

        # deal with CursorKind.UNEXPOSED_REF
        # this shows up when there is an implicit cast between the literal
        # initializer and the storage type of the structure member
        uc = get_child(c, clang.cindex.CursorKind.UNEXPOSED_EXPR) or c

        if (get_child(c, clang.cindex.CursorKind.CXX_BOOL_LITERAL_EXPR) or
                get_child(uc, clang.cindex.CursorKind.CXX_BOOL_LITERAL_EXPR)):
            expected[c.spelling] = OptionType.Bool
        elif (get_child(c, clang.cindex.CursorKind.INTEGER_LITERAL)
              or get_child(uc, clang.cindex.CursorKind.INTEGER_LITERAL)):
            expected[c.spelling] = OptionType.Int
        elif (get_child(c, clang.cindex.CursorKind.FLOATING_LITERAL)
              or get_child(uc, clang.cindex.CursorKind.FLOATING_LITERAL)):
            expected[c.spelling] = OptionType.Float
        else:
            opt_type = get_child(c, clang.cindex.CursorKind.TEMPLATE_REF)
            if opt_type:
                if opt_type.spelling in ["set", "vector", "map"]:
                    expected[c.spelling] = OptionType.Container
                else:
                    assert False, f"Template not supported {opt_type.spelling}"
            else:
                opt_type = get_child(c, clang.cindex.CursorKind.TYPE_REF)
                assert opt_type, (f"Can't find type of {c.spelling}: "
                                  f"{[str(d.kind) for d in children]}")
                if opt_type.spelling in ("std::string",
                                         "std::__cxx11::string"):
                    expected[c.spelling] = OptionType.String
                elif opt_type.spelling == \
                        "class popart::SessionOptions::NumIOTiles":
                    expected[c.spelling] = OptionType.Int
                elif opt_type.spelling.startswith("enum "):
                    expected[c.spelling] = OptionType.Enum
                elif opt_type.spelling.startswith("struct "):
                    expected[c.spelling] = OptionType.Object
                elif opt_type.spelling.startswith("class "):
                    expected[c.spelling] = OptionType.Object
                elif opt_type.spelling == "int64_t":
                    expected[c.spelling] = OptionType.Int
                else:
                    assert False, f"Type not supported {opt_type.spelling}"

    missing = []
    for opt, type in expected.items():
        if opt in options_not_handled:
            continue
        if opt not in handled:
            missing.append(
                f"Option {opt} not handled by PopTorch Type: {str(type)}")
        elif handled[opt] != type:
            missing.append(
                (f"Type mismatch for option {opt}: Popart type {str(type)} "
                 f"PopTorch: {str(handled[opt])}"))
    assert not missing, "\n".join(missing)


index = clang.cindex.Index.create()
session_file = os.path.join(popart_dir, "popart", "sessionoptions.hpp")
tu = index.parse(session_file,
                 args=[
                     "-std=c++14",
                     "-I" + popart_dir,
                     "-DONNX_NAMESPACE=onnx",
                 ])

parse_session_options(tu.cursor)

UnsupportedOps = ["abort", "ctcloss"]

## Implicit cast support
# Casting on all args
CastingOps = [
    "add", "atan2", "bitshift", "clip", "conv", "convtranspose", "div",
    "equal", "gru", "gemm", "greater", "instancenormalization", "less", "lstm",
    "logical_and", "logical_or", "logical_xor", "matmul", "max", "maxroipool",
    "mean", "min", "mod", "mul", "pow", "prelu", "range", "rnn", "scan",
    "sequenceconstruct", "sub", "sum", "groupnormalization", "call",
    "dynamicadd", "dynamicupdate", "dynamiczero", "fmod", "remainder"
]
# Also Einsum, GreaterOrEqual, LessOrEqual

CastingExceptFirstArgsOps = ["where"]
CastingExceptSecondArgsOps = [
    "dequantizelinear", "scatterelements", "scatternd"
]
# Also Pad but only after >= 11
CastingExceptThirdArgsOps = ["roialign"]
CastingExceptFifthArgsOps = ["batchnormalization"]

# Implicit casting ops not in these catagories:
# QLinearConv, QLinearMatMul

# All implicitly casting ops produce an output the same as the promoted type
# except those which always return bools, floats (in onc case) and the following
CastingDifferentOutput = ["sequenceconstruct", "call"]

CastingAlwaysBoolOutput = [
    "equal", "greater", "less", "logical_and", "logical_not", "logical_or",
    "logical_xor"
]

CastingAlwaysFloatOutput = ["dequantizelinear"]

CastingAlwaysIntOutput = ["convinteger", "matmulinteger"]

## Non implicit-casting type support

OutputTypeSameAsFirstInput = [
    "abs", "acos", "acos", "acosh", "asin", "asinh", "atan", "atanh",
    "averagepool", "batchnormalization", "bitwisenot", "ceil", "celu",
    "compress", "concat", "cos", "cosh", "cumsum", "_ctcloss", "depthtospace",
    "det", "detach", "dropout", "dynamicslice", "einsum", "elu", "erf", "exp",
    "expand", "expm1", "flatten", "floor", "fmod", "gather", "gatherelements",
    "gathernd", "gelu", "globalaveragepool", "globallppool", "globalmaxpool",
    "hardmax", "hardsigmoid", "identity", "identityloss", "l1loss", "lrn",
    "leakyrelu", "log", "log1p", "logical_not", "logsoftmax",
    "lpnormalization", "lppool", "maxpool", "maxunpool",
    "meanvariancenormalization", "neg", "nllloss", "nop", "pad", "printtensor",
    "range", "reciprocal", "reducel1", "reducel2", "reducelogsum",
    "reducelogsumexp", "reducemax", "reducemean", "reducemin", "reduceprod",
    "reducesum", "reducesumsquare", "relu", "remainder", "replicatedallreduce",
    "reshape", "resize", "reversesequence", "roialign", "round", "scale",
    "scaledadd", "scatter", "selu", "sequenceerase", "shapeddropout", "shrink",
    "sigmoid", "sign", "sin", "sinh", "slice", "softmax", "softplus",
    "softsign", "spacetodepth", "split", "sqrt", "squeeze", "stringnormalizer",
    "subsample", "tan", "tanh", "thresholdedrelu", "tile", "transpose",
    "unique", "unsqueeze", "upsample"
]

FirstOutputTypeSameAsFirstInputButSecondAlwaysInt = ["topk"]

OutputTypeSameAsThirdInput = ["onehot"]

OutputTypeAlwaysBool = ["isinf", "isnan"]

OutputTypeAlwaysFloat = ["tfidfvectorizer"]

OutputTypeAlwaysInt32 = [
    "argmax", "argmin", "isinf", "isnan", "nonmaxsuppression", "nonzero",
    "shape", "size"
]

OutputTypeAlwaysUint8 = [
    "dynamicquantizelinear", "quantizelinear", "qlinearconv", "qlinearmatmul"
]

OutputTypeAsDtype = [
    "cast", "eyelike", "multinomial", "randomnormal", "randomuniform"
]

OutputTypeAsDtypeOrAsPromoted = ["randomnormallike", "randomuniformlike"]

OutputTypeVariable = [
    "concatfromsequence", "constant", "constantofshape", "loop", "multinomial",
    "sequenceat", "sequentempty", "sequenceinsert ", "splittosequence"
]

MultipleOutputsOps = {"lstm": "2", "split": "num_outputs", "topk": "2"}

CXXTypeToTypeClass = {
    # Scalar integers
    "int64_t": "INT",
    "int": "INT",
    "bool": "INT",
    "unsigned int": "INT",
    "popart::ReductionType": "INT",
    "nonstd::optional<int64_t>": "INT",
    "nonstd::optional<int>": "INT",
    "Attributes::Int": "INT",

    # Floats
    "float": "FLOAT",
    "nonstd::optional<float>": "FLOAT",

    # Non-scalar floats
    "std::vector<float>": "FLOAT_VEC",

    # Non-scalar integers.
    "std::vector<int64_t>": "INT_VEC",
    "nonstd::optional<std::vector<int64_t> >": "INT_VEC",
    "Attributes::Ints": "INT_VEC",

    # String
    "std::string": "STRING"
}


# Convert the raw C++ type parsed from the header into the macro type.
def toType(cxxType):

    cleaned = cxxType.replace("&", "").replace("const", "").strip().rstrip()

    if cleaned in CXXTypeToTypeClass:
        return CXXTypeToTypeClass[cleaned]

    logger.debug("toType: Unknown cxxType=%s / cleaned=%s", cxxType, cleaned)

    # Soft fail as it isn't unexpected for some popart functions to be unsupported right now.
    return "UNKNOWN"


CXX_TYPE_CONV_TABLE = {
    "nonstd::optional<int>": "std::int32_t",
    "nonstd::optional<int64_t>": "std::int32_t",
    "popart::ReductionType": "std::int32_t",
    "nonstd::optional<float>": "float",
    "nonstd::optional<std::vector<int64_t>>": "std::vector<int64_t>",
    "Attributes::Ints": "std::vector<int64_t>",
    "Attributes::Int": "std::int32_t"
}

CXX_NON_CONV_TYPES = [
    "bool", "float", "int64_t", "std::string", "std::vector<int64_t>",
    "unsigned int"
]


# Convert from the popart header types into normal C++ types that can be used by pytorch.
def convertCxxConvert(cxxType_orig):
    cxxType = cxxType_orig.replace("&", "")
    cxxType = cxxType.replace("const ", "const[preserved_space]")
    cxxType = cxxType.replace("unsigned const", "const unsigned")

    # Remove any whitespace but keep "const" and "unsigned" safe
    cxxType = cxxType.replace("const ", "const[preserved_space]")
    cxxType = cxxType.replace("unsigned ", "unsigned[preserved_space]")
    cxxType = "".join(cxxType.split())
    cxxType = cxxType.replace("[preserved_space]", " ")

    if cxxType in CXX_TYPE_CONV_TABLE:
        return CXX_TYPE_CONV_TABLE[cxxType]

    # Most types won't need processing
    if cxxType in CXX_NON_CONV_TYPES:
        return cxxType_orig

    # Handle const
    if cxxType.startswith("const "):
        non_const_type = cxxType[len("const "):]

        if non_const_type in CXX_TYPE_CONV_TABLE:
            # const is dropped for legacy
            return CXX_TYPE_CONV_TABLE[non_const_type]

        if non_const_type in CXX_NON_CONV_TYPES:
            return cxxType_orig

    # Error on unknown types
    print(f"Unknown type: {cxxType}")
    sys.exit(1)


def attrTypeGetter(ty):
    if ty == "INT":
        return "i"

    if ty == "INT_VEC":
        return "is"

    if ty == "FLOAT":
        return "f"

    if ty == "STRING":
        return "s"

    assert False, "Invalid type: " + ty
    return None


def addCastingOptStr(name):
    if name in CastingOps:
        return "ImplicitCast::All"
    if name in CastingExceptFirstArgsOps:
        return "ImplicitCast::ExceptFirst"
    if name in CastingExceptSecondArgsOps:
        return "ImplicitCast::ExceptSecond"
    if name in CastingExceptThirdArgsOps:
        return "ImplicitCast::ExceptThird"
    if name in CastingExceptFifthArgsOps:
        return "ImplicitCast::ExceptFifth"
    return "ImplicitCast::None"


def addOutputTypeStr(name):  # pylint: disable=too-many-return-statements
    if name in CastingAlwaysBoolOutput or name in OutputTypeAlwaysBool:
        return "OutputType::AlwaysBool"
    if name in CastingAlwaysFloatOutput or name in OutputTypeAlwaysFloat:
        return "OutputType::AlwaysFloat"
    if name in CastingAlwaysIntOutput or name in OutputTypeAlwaysInt32:
        return "OutputType::AlwaysInt"
    if any([
            name in n
            for n in (CastingOps, CastingExceptFirstArgsOps,
                      CastingExceptSecondArgsOps, CastingExceptThirdArgsOps)
    ]):
        return "OutputType::AsImplicitCastPromoted"
    if name in OutputTypeSameAsFirstInput:
        return "OutputType::AsFirstInput"
    if name in FirstOutputTypeSameAsFirstInputButSecondAlwaysInt:
        return "OutputType::FirstAsFirstInputSecondAlwaysInt"
    if name in OutputTypeSameAsThirdInput:
        return "OutputType::AsThirdInput"
    if name in OutputTypeAlwaysUint8:
        return "OutputType::AlwaysUint8"
    if name in OutputTypeAsDtype:
        return "OutputType::AsDtype"
    if name in OutputTypeAsDtypeOrAsPromoted:
        return "OutputType::AsDtypeOrAsPromoted"
    if name in OutputTypeVariable:
        return "OutputType::Unknown"
    print(f"Missing type spec for: {name}")
    return "OutputType::Unknown"


macroFile = ""

headerStubs = ""

cxxFile = ""

classes = []
for classname in jsonOutput:
    classes.append(classname)
classes.reverse()

for opset in classes:
    macroFile += "// Ops from %s\n" % opset
    for name in jsonOutput[opset]:
        if name in UnsupportedOps:
            continue

        logger.debug("Generating code for %s::%s", opset, name)
        # Generate the macro
        opDecl = "OP_DECL("

        funcName = name.capitalize()
        opDecl += "popart, " + name + ", " + name

        if opset.startswith("AiOnnxOpset"):
            opDecl += ", AiOnnxOpset10." + name
        else:
            opDecl += ", " + opset + "." + name

        argVector = ""
        bodyArgVector = ""

        earlyExit = True
        args = jsonOutput[opset][name]["args"]
        for arg in args:
            # Skip the first args and also the "name" arg.
            if arg["name"] == "args":
                # Guarantee we are working with an op which takes in popart tensors as 0th argument.
                earlyExit = False
                continue

            macroType = toType(arg["type"])

            if macroType == "UNKNOWN":
                logger.info("Skipping OP: %s"
                            " due to parse failure on %s", name, str(arg))
                earlyExit = True
                break

            argVector += "ARG(" + macroType + "," + arg["name"] + ") "

            if "ReductionType" in arg["type"]:
                bodyArgVector += "BODY_ARG(static_cast<popart::ReductionType>("\
                + arg["name"] + ")) "
            else:
                bodyArgVector += "BODY_ARG(" + arg["name"] + ") "

        if earlyExit:
            continue

        if argVector == "":
            argVector = "NONE"

        if bodyArgVector == "":
            bodyArgVector = "NONE"

        opDecl += ", " + argVector
        opDecl += ", " + bodyArgVector

        macroFile += opDecl + ")\n"

        header = "torch::jit::Node* "

        header += "create" + funcName + "(torch::jit::Graph *graph,  const " \
            "std::vector<torch::jit::Value *>& args"

        cppFile = " torch::jit::Node *new_node = createAndInsertNode(graph, " \
               "symbols::popart::" + name + ", args"

        cppFile += f", {addCastingOptStr(name)}, {addOutputTypeStr(name)}"

        if name in MultipleOutputsOps:
            cppFile += ", %s" % MultipleOutputsOps[name]
        cppFile += ");\n"

        args = jsonOutput[opset][name]["args"]
        for arg in args:
            # Skip the first args
            if arg["name"] == "args":
                continue

            header += "," + convertCxxConvert(arg["type"]) + " " + arg["name"]

            attr = attrTypeGetter(toType(arg["type"]))

            cppFile += "new_node->" + attr + "_(c10::Symbol::fromQualString("\
                "\"attr::" + arg["name"] + "\")," + arg["name"] + ");\n"

        if name in OutputTypeAsDtype:
            cppFile += "setNodeOutputsTypes(new_node, ImplicitCast::None, "
            cppFile += "OutputType::AsDtype);\n"
        if name in OutputTypeAsDtypeOrAsPromoted:
            cppFile += "setNodeOutputsTypes(new_node, ImplicitCast::All, "
            cppFile += "OutputType::AsDtypeOrAsPromoted);\n"

        cppFile += "return new_node;\n"

        cppFile = header + ") {\n" + cppFile + "}"

        header += ");"

        headerStubs += header + "\n"

        cxxFile += cppFile + "\n"

autoComment = """// Copyright (c) 2020 Graphcore Ltd. All rights reserved.
// Auto generated file, do not modify
// Run `python3 PopParse.py to regenerate
// clang-format off
"""

with open(
        os.path.join(_utils.sources_dir(), 'popart_compiler', 'include',
                     'popart_compiler', 'CompilerOperationMacros.inc.hpp'),
        'w') as f:
    print(autoComment, file=f)
    print(macroFile, file=f)

with open(
        os.path.join(_utils.sources_dir(), 'poptorch', 'include', 'poptorch',
                     'CompilerOps.inc.hpp'), 'w') as f:
    print(autoComment, file=f)
    print(headerStubs, file=f)

with open(
        os.path.join(_utils.sources_dir(), 'poptorch', 'source',
                     'CompilerOps.cpp.inc'), 'w') as f:
    print(autoComment, file=f)
    print(cxxFile, file=f)
