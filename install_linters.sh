#!/bin/bash
# Copyright (c) 2020 Graphcore Ltd. All rights reserved.

# realpath doesn't exist on osx
realpath() {
  python3 -c "import os.path; print(os.path.realpath('$1'))"
}

DIR=$(realpath $(dirname $0))

print_usage_and_exit() {
  echo "Usage: $0 --poplar <path_to_poplar> --popart <path_to_popart>"
  exit 1
}

# If some arguments were passed to the script assume this is coming from arc lint: print instructions and exit
if [[ $# -eq 0 ]]
then
  print_usage_and_exit
fi

case "$1" in
  --poplar)
    shift
    POPLAR_PATH=$1
    shift
    ;;
  --popart)
    shift
    POPART_PATH=$1
    shift
    ;;
  *)
    echo "Linters not installed: install them using install_linters.sh then try again"
    print_usage_and_exit
    ;;
esac

torch_path=`python3 -c "import torch; from pathlib import Path; print(Path(torch.__file__).parent, end='')"`
if [ $? -ne 0 ]
then
  echo ""
  echo "Make sure you've activated your pytorch venv and run this script again"
  exit 1
fi
set -e # Stop on error

install_clang_format() {
  CLANG_VERSION=9.0.0
  FOLDER_NAME=clang+llvm-${CLANG_VERSION}-x86_64-${CLANG_PLATFORM}
  ARCHIVE_NAME=${FOLDER_NAME}.tar.xz
  URL=https://releases.llvm.org/${CLANG_VERSION}/${ARCHIVE_NAME}
  INCLUDE_FOLDER=${FOLDER_NAME}/include
  CLANG_INCLUDE_FOLDER=${FOLDER_NAME}/lib/clang/${CLANG_VERSION}/include

  if [ ! -f "${VERSIONED_BINARY}" ]; then
    TMP_DIR=`mktemp -d`
    pushd ${TMP_DIR}
    curl -O $URL
    tar ${TAR_EXTRA_OPTS} -xf ${ARCHIVE_NAME} ${FOLDER_NAME}/bin/clang-format ${FOLDER_NAME}/bin/clang-tidy ${INCLUDE_FOLDER} ${CLANG_INCLUDE_FOLDER}
    mv ${INCLUDE_FOLDER} ${DIR}/.linters/include
    mkdir -p ${DIR}/.linters/include/onnx/
    echo "namespace onnx { enum TensorProto_DataType { TensorProto_DataType_FLOAT }; }" > ${DIR}/.linters/include/onnx/onnx_pb.h
    mv ${CLANG_INCLUDE_FOLDER} ${DIR}/.linters/clang_include
    mv ${FOLDER_NAME}/bin/clang-format ${DIR}/.linters/clang-format
    mv ${FOLDER_NAME}/bin/clang-tidy ${DIR}/.linters/clang-tidy
    chmod +x ${DIR}/.linters/clang-*
    popd
    rm -rf ${TMP_DIR}
  fi
}

cd ${DIR}
mkdir -p .linters

VE=${DIR}/.linters/venv

if [ ! -d ${VE} ]
then
  python3 -m venv ${VE}
fi

source ${VE}/bin/activate
pip install wheel
pip install yapf==0.27.0
pip install cpplint==1.4.4
pip install yml2json
pip install pylint==2.5.3

rm -f .linters/system_includes
if [[ "$OSTYPE" == "linux-gnu" ]]; then
  export CLANG_PLATFORM=linux-gnu-ubuntu-18.04
  export TAR_EXTRA_OPTS="--occurrence"
  install_clang_format
elif [[ "$OSTYPE" == "darwin"* ]]; then
  export CLANG_PLATFORM=darwin-apple
  export TAR_EXTRA_OPTS="--fast-read"
  install_clang_format
  # This include is needed on Mac but breaks things on Linux
  echo " -I.linters/include/c++/v1/ " >> .linters/system_includes
else
  echo "ERROR: '${OSTYPE}' platform not supported"
  exit -1
fi

ln -fs ${VE}/bin/yapf ${DIR}/.linters/yapf
ln -fs ${VE}/bin/yml2json ${DIR}/.linters/yml2json
ln -fs ${VE}/bin/cpplint ${DIR}/.linters/cpplint
ln -fs ${VE}/bin/pylint ${DIR}/.linters/pylint
ln -fs $torch_path .linters/torch
ln -fs ${POPLAR_PATH}/include .linters/poplar_includes
ln -fs ${POPART_PATH}/include .linters/popart_includes
nproc > .linters/num_threads
python3-config --includes >> .linters/system_includes

echo "All linters have been successfully installed"

