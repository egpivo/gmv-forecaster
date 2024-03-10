#!/bin/bash
#
# Build Conda Env
#
# - Parameters
#    - Optional
#       - -c/--conda_env: conda env name; Default: `forecaster`
# - Examples
#       1. Default conda env: forecaster
#           - ./build_conda_env.sh
#       2. New conda env
#           - ./build_conda_env.sh -c forecaster_test
#
# - Caveat
#    - If you don't have `realpath` on Mac, please install via `brew install coreutils`
#
CONDA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
set -a
source "${CONDA_DIR}/conda_env_info.sh"
source "${CONDA_DIR}/utils.sh"
source "${COLOR_MAP_PATH}"
source "${EXIT_CODE_PATH}"
set +a


for ARG in "$@"; do
  shift
  case "${ARG}" in
  "--conda_env") set -- "$@" "-c" ;;
  *) set -- "$@" "${ARG}" ;;
  esac
done

while getopts "d:t:c:*" OPT; do
  case "${OPT}" in
  c)
    CONDA_ENV="${OPTARG}"
    ;;
  *) ;;

  esac
done


build() {
  local CONDA_ENV=$1

  if [ "x${CONDA_ENV}x" == "xx" ]; then
    CONDA_ENV="forecaster"
  fi

  if [ "x${CONDA_HOME}" == "xx" ]; then
    CONDA_HOME=${CONDA_PATH}
  fi

  # Will return `CONDA_ENV_DIR`
  echo -e "${FG_YELLOW}Checking Conda Env: '${CONDA_ENV}'${FG_RESET}"
  find_conda_env_path "${CONDA_ENV}"
  # Try to build the conda env if the error code is captured
  if [ "$?" == "${ERROR_EXITCODE}" ]; then
    PYTHON_VERSION="${DEFAULT_PYTHON_VERSION}"
    retry_to_find_conda_env_path "${CONDA_ENV}" "${PYTHON_VERSION}"
    if [ "$?" == "${ERROR_EXITCODE}" ]; then
      return "${ERROR_EXITCODE}"
    fi
  fi

  initialize_conda
  source activate "${CONDA_ENV}"
  echo -e "${FG_YELLOW}Activating conda env '${CONDA_ENV}'${FG_RESET}"

  # Installation
  echo -e "${FG_YELLOW}Installing package ${FG_RESET}"
  install_python_package "${PACKAGE_BASE_PATH}"
  echo -e "${FG_GREEN}Updated package${FG_RESET}"
  conda deactivate

  return "${SUCCESS_EXITCODE}"
}

build "${CONDA_ENV}"
