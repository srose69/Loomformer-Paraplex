#!/usr/bin/env bash
# ===========================================================================
#  LoomFormer-Paraplex — one-shot installer
#
#  No root. No conda. Git is optional.
#  CUDA toolkit: official NVIDIA .run -> project-local directory, toolkit only.
#  Python: stdlib venv with a virtualenv fallback; PyTorch and nvcc always use
#  the same CUDA line.
#
#  Automatic CUDA profile:
#    * Pascal / Volta present (SM < 7.5): CUDA 12.6 + PyTorch cu126
#    * Turing or newer only (SM >= 7.5): CUDA 13.0 + PyTorch cu130
#
#  Override automatic selection only when probing GPUs is impossible:
#    LOOM_CUDA_LINE=12 ./setup.sh
#    LOOM_CUDA_LINE=13 ./setup.sh
#
#  Other optional overrides:
#    LOOM_INSTALL_DIR, LOOM_REPO_DIR, LOOM_BUILD_JOBS, LOOM_NVCC_THREADS,
#    LOOM_TORCH_VERSION
# ===========================================================================
set -euo pipefail
IFS=$'\n\t'

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; DIM='\033[2m'
RESET='\033[0m'

# ── Project layout ───────────────────────────────────────────────────────────
REPO_URL="https://github.com/srose69/Loomformer-Paraplex.git"
REPO_TAR="https://github.com/srose69/Loomformer-Paraplex/archive/refs/heads/main.tar.gz"
REPO_SENTINEL="loomformer.py"
INSTALL_DIR="${LOOM_INSTALL_DIR:-$HOME/loom}"
SCRIPT_DIR=$(unset CDPATH; cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if [[ -n "${LOOM_REPO_DIR:-}" ]]; then
    REPO_DIR="$LOOM_REPO_DIR"
elif [[ -f "$SCRIPT_DIR/$REPO_SENTINEL" ]]; then
    # Normal checked-out-repository use: update the checkout containing setup.sh.
    REPO_DIR="$SCRIPT_DIR"
else
    # One-shot use of a separately downloaded setup.sh.
    REPO_DIR="$INSTALL_DIR/Loomformer-Paraplex"
fi
VENV_DIR="$INSTALL_DIR/venv"
ENV_FILE="$INSTALL_DIR/loomformer-env.sh"
VALIDATION_REPORT="$INSTALL_DIR/validation-report.json"

# CUDA 13 is the normal profile. CUDA 12.6 is retained because CUDA 13 removed
# Maxwell, Pascal and Volta, while PyTorch's cu126 build still carries them.
CUDA13_VER="13.0.2"
CUDA13_DRIVER_VER="580.95.05"
CUDA13_RUN_MD5="3f092554675f004250d4dfc1d6c3acc9"

CUDA12_VER="12.6.3"
CUDA12_DRIVER_VER="560.35.05"
CUDA12_RUN_MD5="29d297908c72b810c9ceaa5177142abd"

TORCH_VER="${LOOM_TORCH_VERSION:-2.12.1}"
PYTHON_MIN_MAJOR=3
PYTHON_MIN_MINOR=10

# Filled by select_cuda_profile().
CUDA_LINE=""
CUDA_VER=""
CUDA_DRIVER_VER=""
CUDA_RUN_MD5=""
CUDA_RUN_URL=""
CUDA_DIR=""
TORCH_FLAVOR=""
TORCH_IDX=""
TORCH_CUDA_ARCH_LIST=""
FLASH_ATTN_CUDA_ARCHS=""
INSTALL_FLASH_ATTN=0
BUILD_JOBS=1
NVCC_BUILD_THREADS=2

# ── Probe state ──────────────────────────────────────────────────────────────
HAS_GIT=0
HAS_GCC=0
HAS_CXX=0
HAS_CURL=0
HAS_WGET=0
HAS_NVIDIA_SMI=0
PYTHON_OK=0
PYTHON_BIN=""
GPU_CAPS=()

# ═══════════════════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════════════════
log()  { echo -e "${GREEN}[✓]${RESET} $*"; }
info() { echo -e "${CYAN}[i]${RESET} $*"; }
warn() { echo -e "${YELLOW}[!]${RESET} $*"; }
die()  { echo -e "${RED}[✗]${RESET} $*" >&2; exit 1; }
step() { echo -e "\n${BOLD}${BLUE}━━━ $* ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"; }

pip_run() {
    "$VENV_DIR/bin/python3" -m pip "$@"
}

download_file() {
    local url="$1"
    local dest="$2"
    mkdir -p "$(dirname "$dest")"
    if (( HAS_CURL )); then
        curl -fL# --retry 5 --retry-delay 2 --continue-at - -o "$dest" "$url"
    elif (( HAS_WGET )); then
        wget --show-progress --tries=5 --continue -O "$dest" "$url"
    else
        die "Neither curl nor wget is available."
    fi
}

version_ge() {
    # Numeric major.minor.patch comparison. Usage: version_ge ACTUAL REQUIRED
    local actual="$1"
    local required="$2"
    local a_major a_minor a_patch r_major r_minor r_patch
    IFS=. read -r a_major a_minor a_patch _ <<< "$actual"
    IFS=. read -r r_major r_minor r_patch _ <<< "$required"
    a_minor="${a_minor:-0}"; a_patch="${a_patch:-0}"
    r_minor="${r_minor:-0}"; r_patch="${r_patch:-0}"
    [[ "$a_major" =~ ^[0-9]+$ && "$a_minor" =~ ^[0-9]+$ &&
       "$a_patch" =~ ^[0-9]+$ ]] || return 1
    [[ "$r_major" =~ ^[0-9]+$ && "$r_minor" =~ ^[0-9]+$ &&
       "$r_patch" =~ ^[0-9]+$ ]] || return 1
    (( 10#$a_major > 10#$r_major ||
       (10#$a_major == 10#$r_major && 10#$a_minor > 10#$r_minor) ||
       (10#$a_major == 10#$r_major && 10#$a_minor == 10#$r_minor &&
        10#$a_patch >= 10#$r_patch) ))
}

safe_remove_temp_tree() {
    local path="${1:-}"
    local temp_root="${TMPDIR:-/tmp}"
    temp_root="${temp_root%/}"
    [[ -n "$temp_root" ]] || temp_root="/tmp"
    case "$path" in
        "$temp_root"/loomformer-update.*)
            [[ -d "$path" ]] && rm -rf -- "$path"
            ;;
        "")
            ;;
        *)
            warn "Refusing to remove unexpected temporary path: $path"
            ;;
    esac
}

safe_remove_build_log() {
    local path="${1:-}"
    local temp_root="${TMPDIR:-/tmp}"
    temp_root="${temp_root%/}"
    [[ -n "$temp_root" ]] || temp_root="/tmp"
    case "$path" in
        "$temp_root"/loomformer-flash-attn.*.log)
            [[ -f "$path" ]] && rm -f -- "$path"
            ;;
        "")
            ;;
        *)
            warn "Refusing to remove unexpected build log: $path"
            ;;
    esac
}

filter_flash_attn_build_output() {
    local build_log="$1"
    local line=""
    local progress_active=0
    local current=0
    local total=0

    exec 3> "$build_log" || return 1
    while IFS= read -r line; do
        printf '%s\n' "$line" >&3 || {
            exec 3>&-
            return 1
        }

        if [[ "$line" =~ \[([0-9]+)/([0-9]+)\] ]]; then
            current="${BASH_REMATCH[1]}"
            total="${BASH_REMATCH[2]}"
            if [[ -t 1 ]]; then
                printf '\r\033[2K  [flash-attn] [%s/%s]' "$current" "$total"
            else
                printf '  [flash-attn] [%s/%s]\n' "$current" "$total"
            fi
            progress_active=1
        elif [[ "$line" == *"Precompiled wheel not found. Building from source"* ]]; then
            (( progress_active )) && [[ -t 1 ]] && printf '\n'
            info "No compatible prebuilt FlashAttention wheel; compiling locally."
            progress_active=0
        elif [[ "$line" == *"Successfully built flash-attn"* ]]; then
            (( progress_active )) && [[ -t 1 ]] && printf '\n'
            log "FlashAttention wheel built."
            progress_active=0
        fi
    done

    (( progress_active )) && [[ -t 1 ]] && printf '\n'
    exec 3>&-
    return 0
}

set_cuda_env() {
    [[ -n "$CUDA_DIR" ]] || die "Internal error: CUDA profile is not selected."
    export CUDA_HOME="$CUDA_DIR"
    export PATH="$CUDA_DIR/bin:${PATH}"
    export LD_LIBRARY_PATH="$CUDA_DIR/lib64:${LD_LIBRARY_PATH:-}"
    export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"
    export FLASH_ATTN_CUDA_ARCHS="$FLASH_ATTN_CUDA_ARCHS"
    export MAX_JOBS="$BUILD_JOBS"
    export NVCC_THREADS="$NVCC_BUILD_THREADS"
}

repo_exists() {
    [[ -f "$REPO_DIR/$REPO_SENTINEL" ]]
}

# ═══════════════════════════════════════════════════════════════════════════
#  Banner
# ═══════════════════════════════════════════════════════════════════════════
print_logo() {
    echo -e "${CYAN}"
    cat << 'LOGO'
  ██╗      ██████╗  ██████╗ ███╗   ███╗███████╗ ██████╗ ██████╗ ███╗   ███╗███████╗██████╗
  ██║     ██╔═══██╗██╔═══██╗████╗ ████║██╔════╝██╔═══██╗██╔══██╗████╗ ████║██╔════╝██╔══██╗
  ██║     ██║   ██║██║   ██║██╔████╔██║█████╗  ██║   ██║██████╔╝██╔████╔██║█████╗  ██████╔╝
  ██║     ██║   ██║██║   ██║██║╚██╔╝██║██╔══╝  ██║   ██║██╔══██╗██║╚██╔╝██║██╔══╝  ██╔══██╗
  ███████╗╚██████╔╝╚██████╔╝██║ ╚═╝ ██║██║     ╚██████╔╝██║  ██║██║ ╚═╝ ██║███████╗██║  ██║
  ╚══════╝ ╚═════╝  ╚═════╝ ╚═╝     ╚═╝╚═╝      ╚═════╝ ╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝
LOGO
    echo -e "${DIM}       Paraplex language model · Pascal-to-Blackwell CUDA paths${RESET}"
    echo -e "${DIM}       https://github.com/srose69/Loomformer-Paraplex${RESET}\n"
}

# ═══════════════════════════════════════════════════════════════════════════
#  GPU and toolchain selection
# ═══════════════════════════════════════════════════════════════════════════
probe_gpu_caps() {
    GPU_CAPS=()
    (( HAS_NVIDIA_SMI )) || return 0

    local cap
    while IFS= read -r cap; do
        cap="${cap//[[:space:]]/}"
        [[ "$cap" =~ ^[0-9]+\.[0-9]+$ ]] || continue
        GPU_CAPS+=("$cap")
    done < <(nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null || true)
}

select_cuda_profile() {
    local requested="${LOOM_CUDA_LINE:-}"
    local use_cuda12=0
    local has_ampere=0
    local has_blackwell=0
    local cap major minor sm
    local -a unique_caps=()
    local -a flash_archs=()

    TORCH_CUDA_ARCH_LIST=""
    FLASH_ATTN_CUDA_ARCHS=""
    INSTALL_FLASH_ATTN=0

    if [[ -n "$requested" && "$requested" != "12" && "$requested" != "13" ]]; then
        die "LOOM_CUDA_LINE must be 12 or 13, got '$requested'."
    fi

    if [[ -z "$requested" ]]; then
        ((${#GPU_CAPS[@]} > 0)) || die \
            "Could not detect NVIDIA compute capability. Ensure nvidia-smi works, or set LOOM_CUDA_LINE=12/13 explicitly."
        for cap in "${GPU_CAPS[@]}"; do
            IFS=. read -r major minor <<< "$cap"
            sm=$((10#$major * 10 + 10#$minor))
            (( sm >= 60 )) || die \
                "GPU compute capability $cap is below LoomFormer's Pascal baseline (SM 6.0)."
            (( sm < 75 )) && use_cuda12=1
        done
        if (( use_cuda12 )); then
            requested=12
        else
            requested=13
        fi
    fi

    if [[ "$requested" == "12" ]]; then
        CUDA_LINE="12"
        CUDA_VER="$CUDA12_VER"
        CUDA_DRIVER_VER="$CUDA12_DRIVER_VER"
        CUDA_RUN_MD5="$CUDA12_RUN_MD5"
        TORCH_FLAVOR="cu126"
    else
        CUDA_LINE="13"
        CUDA_VER="$CUDA13_VER"
        CUDA_DRIVER_VER="$CUDA13_DRIVER_VER"
        CUDA_RUN_MD5="$CUDA13_RUN_MD5"
        TORCH_FLAVOR="cu130"
    fi

    CUDA_DIR="$INSTALL_DIR/cuda-${CUDA_VER%.*}"
    CUDA_RUN_URL="https://developer.download.nvidia.com/compute/cuda/${CUDA_VER}/local_installers/cuda_${CUDA_VER}_${CUDA_DRIVER_VER}_linux.run"
    TORCH_IDX="https://download.pytorch.org/whl/${TORCH_FLAVOR}"

    # Build project extensions only for the GPUs visible during setup. This
    # avoids compiling every architecture PyTorch knows about.
    for cap in "${GPU_CAPS[@]}"; do
        local seen=0 existing
        for existing in "${unique_caps[@]:-}"; do
            [[ "$existing" == "$cap" ]] && seen=1
        done
        (( seen )) || unique_caps+=("$cap")

        IFS=. read -r major minor <<< "$cap"
        sm=$((10#$major * 10 + 10#$minor))
        (( sm >= 60 )) || die \
            "GPU compute capability $cap is below LoomFormer's Pascal baseline (SM 6.0)."
        (( sm < 75 )) && use_cuda12=1
        (( sm >= 80 )) && has_ampere=1
        (( sm >= 100 )) && has_blackwell=1

        # FlashAttention owns its CUDA target selection and does not consume
        # TORCH_CUDA_ARCH_LIST. Its supported source targets represent GPU
        # families: sm_80 covers Ampere/Ada, followed by Hopper and the
        # Blackwell/Thor families.
        local flash_arch=""
        if (( sm >= 80 && sm < 90 )); then
            flash_arch="80"
        elif (( sm >= 90 && sm < 100 )); then
            flash_arch="90"
        elif (( sm >= 100 && sm < 110 )); then
            flash_arch="100"
        elif (( sm >= 110 && sm < 120 )); then
            flash_arch="110"
        elif (( sm >= 120 )); then
            flash_arch="120"
        fi
        if [[ -n "$flash_arch" ]]; then
            local flash_seen=0
            for existing in "${flash_archs[@]:-}"; do
                [[ "$existing" == "$flash_arch" ]] && flash_seen=1
            done
            (( flash_seen )) || flash_archs+=("$flash_arch")
        fi
    done

    if [[ "$requested" == "13" ]] && (( use_cuda12 )); then
        die "CUDA 13 cannot target visible Pascal/Volta GPU(s). Remove LOOM_CUDA_LINE=13 or use separate environments."
    fi
    if [[ "$requested" == "12" ]] && (( has_blackwell )); then
        die "The cu126 compatibility profile cannot target visible Blackwell GPU(s). Use CUDA 13, or separate CUDA 12.6 and CUDA 13.0 environments for a mixed legacy/Blackwell fleet."
    fi

    if ((${#unique_caps[@]} > 0)); then
        local joined=""
        for cap in "${unique_caps[@]}"; do
            joined+="${joined:+;}${cap}"
        done
        TORCH_CUDA_ARCH_LIST="$joined"
    elif [[ "$CUDA_LINE" == "12" ]]; then
        TORCH_CUDA_ARCH_LIST="6.0;6.1"
    else
        # Explicit override on a head/login node: compile a forward-compatible
        # modern baseline. A target node should rerun package installation to
        # compile exact local architectures.
        TORCH_CUDA_ARCH_LIST="8.0+PTX"
    fi

    if ((${#flash_archs[@]} > 0)); then
        local flash_joined=""
        for flash_arch in "${flash_archs[@]}"; do
            flash_joined+="${flash_joined:+;}${flash_arch}"
        done
        FLASH_ATTN_CUDA_ARCHS="$flash_joined"
    elif [[ "$CUDA_LINE" == "13" ]]; then
        # A login/head-node install gets an Ampere-family cubin plus PTX.
        # Installs performed on a GPU node are narrowed to its exact family.
        FLASH_ATTN_CUDA_ARCHS="80"
    fi

    INSTALL_FLASH_ATTN="$has_ampere"
    if ((${#GPU_CAPS[@]} == 0)) && [[ "$CUDA_LINE" == "13" ]]; then
        # Explicit CUDA 13 override normally targets a modern remote GPU.
        INSTALL_FLASH_ATTN=1
    fi

    calculate_build_jobs
}

calculate_build_jobs() {
    local requested_nvcc_threads="${LOOM_NVCC_THREADS:-${NVCC_THREADS:-2}}"
    [[ "$requested_nvcc_threads" =~ ^[1-9][0-9]*$ ]] || die \
        "LOOM_NVCC_THREADS/NVCC_THREADS must be a positive integer."
    NVCC_BUILD_THREADS="$requested_nvcc_threads"

    if [[ -n "${LOOM_BUILD_JOBS:-}" ]]; then
        [[ "$LOOM_BUILD_JOBS" =~ ^[1-9][0-9]*$ ]] || die \
            "LOOM_BUILD_JOBS must be a positive integer."
        BUILD_JOBS="$LOOM_BUILD_JOBS"
        return
    fi

    local cpu_count=1
    local mem_kib=0
    local cpu_jobs=1
    local memory_jobs=1
    command -v nproc &>/dev/null && cpu_count=$(nproc)
    cpu_jobs=$((cpu_count / 2))
    (( cpu_jobs >= 1 )) || cpu_jobs=1
    [[ -r /proc/meminfo ]] && mem_kib=$(awk '/^MemAvailable:/ {print $2; exit}' /proc/meminfo)
    if [[ "$mem_kib" =~ ^[0-9]+$ ]] && (( mem_kib > 0 )); then
        # Match FlashAttention's own worst-case estimate: each Ninja job may
        # run NVCC_THREADS compiler threads at roughly 5 GiB apiece.
        memory_jobs=$((mem_kib / 1024 / 1024 / (5 * NVCC_BUILD_THREADS)))
        (( memory_jobs >= 1 )) || memory_jobs=1
    fi
    BUILD_JOBS="$cpu_jobs"
    (( BUILD_JOBS > memory_jobs )) && BUILD_JOBS="$memory_jobs"
    (( BUILD_JOBS >= 1 )) || BUILD_JOBS=1
}

check_driver_compatibility() {
    ((${#GPU_CAPS[@]} > 0)) || return 0
    [[ "${LOOM_SKIP_DRIVER_CHECK:-0}" == "1" ]] && return 0

    local driver_version required_driver
    driver_version=$(nvidia-smi --query-gpu=driver_version \
        --format=csv,noheader,nounits 2>/dev/null | head -n 1)
    driver_version="${driver_version//[[:space:]]/}"
    if [[ "$CUDA_LINE" == "12" ]]; then
        # CUDA 12.x minor-version compatibility floor on Linux.
        required_driver="525.60.13"
    else
        # CUDA 13.x minor-version compatibility floor on Linux.
        required_driver="580.65.06"
    fi
    if [[ -z "$driver_version" ]]; then
        warn "Could not read the NVIDIA driver version; continuing."
        return 0
    fi
    version_ge "$driver_version" "$required_driver" || die \
        "NVIDIA driver $driver_version is too old for CUDA ${CUDA_VER%.*} minor-version compatibility; $required_driver or newer is required."
}

probe_tools() {
    step "Scanning environment"

    command -v curl &>/dev/null && {
        HAS_CURL=1
        log "curl  found: $(curl --version | head -n 1)"
    }
    command -v wget &>/dev/null && {
        HAS_WGET=1
        log "wget  found: $(wget --version 2>&1 | head -n 1)"
    }
    (( HAS_CURL + HAS_WGET > 0 )) || die "Neither curl nor wget is available."

    if command -v git &>/dev/null; then
        HAS_GIT=1
        log "git   found: $(git --version)"
    else
        warn "git not found — repository tarball fallback will be used."
    fi

    if command -v gcc &>/dev/null; then
        HAS_GCC=1
        log "gcc   found: $(gcc --version | head -n 1)"
    else
        warn "gcc not found — CUDA extension compilation will fail."
    fi

    if command -v g++ &>/dev/null; then
        HAS_CXX=1
        log "g++   found: $(g++ --version | head -n 1)"
    else
        warn "g++ not found — CUDA extension compilation will fail."
    fi

    if command -v nvidia-smi &>/dev/null; then
        HAS_NVIDIA_SMI=1
        log "NVIDIA driver found."
    else
        warn "nvidia-smi not found."
    fi

    local py major minor
    for py in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
        command -v "$py" &>/dev/null || continue
        major=$("$py" -c 'import sys; print(sys.version_info.major)' 2>/dev/null) || continue
        minor=$("$py" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null) || continue
        if (( major > PYTHON_MIN_MAJOR ||
              (major == PYTHON_MIN_MAJOR && minor >= PYTHON_MIN_MINOR) )); then
            PYTHON_BIN="$py"
            PYTHON_OK=1
            log "python found: $("$py" --version) -> $(command -v "$py")"
            break
        fi
    done
    (( PYTHON_OK )) || die \
        "No Python >= ${PYTHON_MIN_MAJOR}.${PYTHON_MIN_MINOR} found. Install Python first."

    probe_gpu_caps
    select_cuda_profile
    check_driver_compatibility

    if ((${#GPU_CAPS[@]} > 0)); then
        info "Visible GPU compute capabilities: ${GPU_CAPS[*]}"
    fi
    log "Selected CUDA $CUDA_VER / PyTorch $TORCH_FLAVOR / architectures $TORCH_CUDA_ARCH_LIST"
    info "Native build parallelism: MAX_JOBS=$BUILD_JOBS"

    if [[ -x "$CUDA_DIR/bin/nvcc" ]]; then
        log "local nvcc: $("$CUDA_DIR/bin/nvcc" --version | grep release | tr -d '\n')"
    elif command -v nvcc &>/dev/null; then
        warn "System nvcc exists, but setup will use an isolated CUDA $CUDA_VER toolkit at $CUDA_DIR."
    else
        info "CUDA $CUDA_VER toolkit will be installed at $CUDA_DIR."
    fi

    if repo_exists; then
        info "Repository already present at $REPO_DIR."
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
#  Menu
# ═══════════════════════════════════════════════════════════════════════════
menu() {
    echo
    echo -e "${BOLD}What would you like to do?${RESET}"
    echo -e "  ${BOLD}1${RESET}) Full install  ${DIM}(repo + local CUDA + venv + packages + validation)${RESET}"
    echo -e "  ${BOLD}2${RESET}) Check environment only  ${DIM}(nothing is installed or changed)${RESET}"
    echo -e "  ${BOLD}3${RESET}) Update repository  ${DIM}(tracked upstream content only)${RESET}"
    echo -e "  ${BOLD}4${RESET}) Reinstall Python packages  ${DIM}(requires the selected local CUDA toolkit)${RESET}"
    echo -e "  ${BOLD}5${RESET}) Exit"
    echo

    local choice
    IFS= read -rp "$(echo -e "${YELLOW}→${RESET} Choice [1-5]: ")" choice </dev/tty || {
        echo
        die "Cannot read /dev/tty. Use a TTY, or pipe choice 1 to run a full install."
    }
    echo
    case "$choice" in
        1) do_full_install ;;
        2) do_check_only ;;
        3) do_update_repo ;;
        4) do_packages_only ;;
        5) echo -e "${DIM}Bye.${RESET}"; exit 0 ;;
        *) warn "Unknown option '$choice'"; menu ;;
    esac
}

do_check_only() {
    step "Environment summary"
    local ok=1
    local downloader_text git_text python_text compiler_text
    if (( HAS_CURL + HAS_WGET > 0 )); then
        downloader_text="${GREEN}OK${RESET}"
    else
        downloader_text="${RED}MISSING${RESET}"
        ok=0
    fi
    if (( HAS_GIT )); then
        git_text="${GREEN}OK${RESET}"
    else
        git_text="${YELLOW}missing (tar fallback available)${RESET}"
    fi
    if (( PYTHON_OK )); then
        python_text="${GREEN}$PYTHON_BIN${RESET}"
    else
        python_text="${RED}MISSING${RESET}"
        ok=0
    fi
    if (( HAS_GCC && HAS_CXX )); then
        compiler_text="${GREEN}OK${RESET}"
    else
        compiler_text="${RED}MISSING${RESET}"
        ok=0
    fi
    echo -e "  downloader : $downloader_text"
    echo -e "  git        : $git_text"
    echo -e "  python     : $python_text"
    echo -e "  gcc/g++    : $compiler_text"
    echo -e "  GPU caps   : ${CYAN}${GPU_CAPS[*]:-explicit override / unavailable}${RESET}"
    echo -e "  CUDA       : ${CYAN}$CUDA_VER -> $CUDA_DIR${RESET}"
    echo -e "  PyTorch    : ${CYAN}$TORCH_VER ($TORCH_FLAVOR)${RESET}"
    echo -e "  venv       : $( [[ -x "$VENV_DIR/bin/python3" ]] && echo "${GREEN}$VENV_DIR${RESET}" || echo "${YELLOW}not created${RESET}")"
    echo -e "  repo       : $( repo_exists && echo "${GREEN}$REPO_DIR${RESET}" || echo "${YELLOW}not cloned${RESET}")"
    echo
    if (( ok )); then
        log "Critical host tools are available."
    else
        warn "Install the missing compiler/Python tools before full setup."
    fi
}

# ═══════════════════════════════════════════════════════════════════════════
#  Repository fetch and safe update
# ═══════════════════════════════════════════════════════════════════════════

# These are local runtime/user-data roots. Update never overwrites anything
# under them, even if a similarly named path is added upstream later.
PRESERVE_DIRS=(
    alt alt5 alt6 alts
    ckpt checkpoints
    cuda datasets logs outputs runs tmp
    venv .venv
)
PRESERVE_GLOBS=(
    "*.pt" "*.pth" "*.bin" "*.log" "*.safetensors"
    "*.parquet" "*.arrow" "*.feather"
)

path_is_preserved() {
    local rel="$1"
    local item pattern
    for item in "${PRESERVE_DIRS[@]}"; do
        [[ "$rel" == "$item" || "$rel" == "$item/"* ]] && return 0
    done
    for pattern in "${PRESERVE_GLOBS[@]}"; do
        # shellcheck disable=SC2254
        case "$rel" in $pattern) return 0 ;; esac
    done
    return 1
}

fetch_repo() {
    if repo_exists; then
        info "Repository already exists; synchronizing it first."
        do_update_repo
        return 0
    fi

    [[ ! -e "$REPO_DIR" ]] || die \
        "$REPO_DIR exists but is not a Loomformer checkout. Refusing to overwrite it."
    mkdir -p "$(dirname "$REPO_DIR")"

    if (( HAS_GIT )); then
        info "Cloning repository."
        git clone --depth 1 "$REPO_URL" "$REPO_DIR"
    else
        local archive="$INSTALL_DIR/.loomformer-main.tar.gz"
        local extract_root="$INSTALL_DIR/.repo-extract"
        [[ ! -e "$extract_root" ]] || die \
            "Temporary extraction path already exists: $extract_root"
        download_file "$REPO_TAR" "$archive"
        mkdir -p "$extract_root"
        tar -xzf "$archive" --no-same-owner --no-same-permissions \
            --strip-components=1 -C "$extract_root"
        [[ -f "$extract_root/$REPO_SENTINEL" ]] || die \
            "Downloaded archive does not contain $REPO_SENTINEL."
        mv "$extract_root" "$REPO_DIR"
        rm -f -- "$archive"
    fi
    repo_exists || die "Repository fetch did not produce $REPO_DIR/$REPO_SENTINEL."
    log "Repository ready at $REPO_DIR."
}

do_update_repo() {
    step "Updating repository"
    repo_exists || die "Repository not found at $REPO_DIR. Run full install first."

    if [[ -d "$REPO_DIR/.git" ]] && (( HAS_GIT )); then
        local origin
        origin=$(git -C "$REPO_DIR" remote get-url origin 2>/dev/null || true)
        [[ "$origin" == "$REPO_URL" ||
           "$origin" == "${REPO_URL%.git}" ||
           "$origin" == "git@github.com:srose69/Loomformer-Paraplex.git" ]] || die \
            "Unexpected git origin '$origin' in $REPO_DIR. Refusing to pull a different repository."

        info "Running git pull --ff-only. Git will touch tracked paths only."
        if ! git -C "$REPO_DIR" pull --ff-only; then
            die "git pull failed. Local tracked changes were left intact; resolve them explicitly and retry."
        fi
        log "Repository updated via fast-forward git pull. Untracked user data was untouched."
        return 0
    fi

    # Tarball installs have no previous tracked-file index. Overlay only regular
    # files/symlinks that exist in the new upstream archive. Never delete a
    # local-only path and never replace a directory with a file.
    local update_tmp
    update_tmp=$(mktemp -d "${TMPDIR:-/tmp}/loomformer-update.XXXXXXXX")
    trap 'safe_remove_temp_tree "${update_tmp:-}"' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM

    local archive="$update_tmp/repo.tar.gz"
    local fresh="$update_tmp/repo"
    download_file "$REPO_TAR" "$archive"
    mkdir -p "$fresh"
    tar -xzf "$archive" --no-same-owner --no-same-permissions \
        --strip-components=1 -C "$fresh"
    [[ -f "$fresh/$REPO_SENTINEL" ]] || die \
        "Downloaded archive does not contain $REPO_SENTINEL."

    local changed=0 unchanged=0 preserved=0 conflicts=0
    local source rel target parent component parent_cursor symlink_parent
    local -a rel_components=()
    while IFS= read -r -d '' source; do
        rel="${source#"$fresh"/}"
        if path_is_preserved "$rel"; then
            ((preserved += 1))
            continue
        fi

        target="$REPO_DIR/$rel"
        parent=$(dirname "$target")
        symlink_parent=0
        parent_cursor="$REPO_DIR"
        IFS=/ read -r -a rel_components <<< "$rel"
        for component in "${rel_components[@]:0:${#rel_components[@]}-1}"; do
            parent_cursor="$parent_cursor/$component"
            if [[ -L "$parent_cursor" ]]; then
                symlink_parent=1
                break
            fi
        done
        if (( symlink_parent )); then
            warn "Update conflict: parent of '$rel' is a local symlink; skipped."
            ((conflicts += 1))
            continue
        fi
        if [[ -e "$target" && -d "$target" && ! -L "$target" ]]; then
            warn "Update conflict: upstream file '$rel' is a local directory; skipped."
            ((conflicts += 1))
            continue
        fi
        if [[ -e "$parent" && ! -d "$parent" ]]; then
            warn "Update conflict: parent of '$rel' is a local file; skipped."
            ((conflicts += 1))
            continue
        fi

        mkdir -p "$parent"
        if [[ -L "$source" ]]; then
            local link_value current_link=""
            link_value=$(readlink "$source")
            [[ -L "$target" ]] && current_link=$(readlink "$target")
            if [[ "$current_link" == "$link_value" ]]; then
                ((unchanged += 1))
            elif [[ -e "$target" && ! -L "$target" ]]; then
                warn "Update conflict: upstream symlink '$rel' is a local file; skipped."
                ((conflicts += 1))
            else
                ln -sfn -- "$link_value" "$target"
                ((changed += 1))
            fi
        elif [[ -L "$target" ]]; then
            warn "Update conflict: upstream file '$rel' is a local symlink; skipped."
            ((conflicts += 1))
        elif [[ -f "$target" ]] && cmp -s "$source" "$target"; then
            ((unchanged += 1))
        else
            cp -p -- "$source" "$target"
            ((changed += 1))
        fi
    done < <(find "$fresh" \( -type f -o -type l \) -print0)

    safe_remove_temp_tree "$update_tmp"
    trap - EXIT INT TERM
    info "Update overlay: changed=$changed unchanged=$unchanged preserved=$preserved conflicts=$conflicts"
    (( conflicts == 0 )) || die \
        "Repository update completed with $conflicts path conflict(s). Nothing local was deleted; resolve the reported paths."
    log "Repository synchronized from the upstream archive. Local-only files and user directories were untouched."
}

# ═══════════════════════════════════════════════════════════════════════════
#  Local CUDA toolkit
# ═══════════════════════════════════════════════════════════════════════════
verify_local_cuda() {
    [[ -x "$CUDA_DIR/bin/nvcc" ]] || return 1
    local detected
    detected=$("$CUDA_DIR/bin/nvcc" --version |
        sed -n 's/.*release \([0-9][0-9.]*\).*/\1/p' |
        head -n 1)
    [[ "$detected" == "${CUDA_VER%.*}" ]]
}

install_cuda_runfile() {
    if verify_local_cuda; then
        set_cuda_env
        log "CUDA toolkit already present: $("$CUDA_DIR/bin/nvcc" --version | grep release | tr -d '\n')"
        return 0
    fi

    if [[ -d "$CUDA_DIR" ]] && find "$CUDA_DIR" -mindepth 1 -print -quit | grep -q .; then
        die "$CUDA_DIR is non-empty but does not contain the expected CUDA ${CUDA_VER%.*} nvcc. Refusing to delete or overwrite it."
    fi

    local downloads="$INSTALL_DIR/.downloads"
    local runfile
    runfile="$downloads/$(basename "$CUDA_RUN_URL")"
    mkdir -p "$downloads"
    info "Downloading CUDA $CUDA_VER toolkit-only installer (~4.5 GB)."
    download_file "$CUDA_RUN_URL" "$runfile"

    info "Verifying NVIDIA-published MD5 checksum."
    local actual_md5
    actual_md5=$(md5sum "$runfile" | awk '{print $1}')
    [[ "$actual_md5" == "$CUDA_RUN_MD5" ]] || die \
        "CUDA installer checksum mismatch: expected $CUDA_RUN_MD5, got $actual_md5. The retained file is $runfile."

    if [[ -n "${LOOM_CUDA_SHA256:-}" ]]; then
        info "Verifying caller-supplied SHA256 checksum."
        local actual_sha256
        actual_sha256=$(sha256sum "$runfile" | awk '{print $1}')
        [[ "$actual_sha256" == "$LOOM_CUDA_SHA256" ]] || die \
            "CUDA installer SHA256 mismatch."
    fi

    mkdir -p "$CUDA_DIR"
    chmod +x "$runfile"
    info "Installing CUDA toolkit to $CUDA_DIR (no driver, no root)."
    sh "$runfile" \
        --silent \
        --toolkit \
        --toolkitpath="$CUDA_DIR" \
        --no-man-page \
        --override

    verify_local_cuda || die \
        "CUDA installer finished, but expected nvcc ${CUDA_VER%.*} was not found in $CUDA_DIR."
    set_cuda_env
    log "nvcc installed: $("$CUDA_DIR/bin/nvcc" --version | grep release | tr -d '\n')"
}

# ═══════════════════════════════════════════════════════════════════════════
#  Python environment and packages
# ═══════════════════════════════════════════════════════════════════════════
install_venv() {
    if [[ -x "$VENV_DIR/bin/python3" ]] &&
       "$VENV_DIR/bin/python3" -m pip --version &>/dev/null; then
        info "Python venv already healthy at $VENV_DIR."
        return 0
    fi

    backup_incomplete_venv() {
        [[ -e "$VENV_DIR" ]] || return 0
        local stamp backup suffix
        stamp=$(date +%Y%m%d_%H%M%S)
        backup="${VENV_DIR}.incomplete.${stamp}"
        suffix=0
        while [[ -e "$backup" ]]; do
            suffix=$((suffix + 1))
            backup="${VENV_DIR}.incomplete.${stamp}.${suffix}"
        done
        warn "Preserving incomplete venv as $backup."
        mv -- "$VENV_DIR" "$backup"
    }

    if [[ -e "$VENV_DIR" ]]; then
        backup_incomplete_venv
    fi

    if "$PYTHON_BIN" -m ensurepip --version &>/dev/null; then
        info "Creating Python venv at $VENV_DIR."
        if "$PYTHON_BIN" -m venv "$VENV_DIR" >/dev/null 2>&1 &&
           "$VENV_DIR/bin/python3" -m pip --version &>/dev/null; then
            log "Venv created: $("$VENV_DIR/bin/python3" --version)"
            return 0
        fi
        backup_incomplete_venv
    fi

    warn "python -m venv is unavailable or did not produce a healthy environment; falling back to virtualenv."

    if ! "$PYTHON_BIN" -m virtualenv --version &>/dev/null; then
        "$PYTHON_BIN" -m pip --version &>/dev/null || die \
            "$PYTHON_BIN has neither a working stdlib venv nor pip. Install python3-pip or a matching python-venv package and retry."
        info "Installing virtualenv into the current user's Python site."
        "$PYTHON_BIN" -m pip install --user virtualenv
    fi

    info "Creating Python environment with virtualenv at $VENV_DIR."
    if "$PYTHON_BIN" -m virtualenv "$VENV_DIR" &&
       [[ -x "$VENV_DIR/bin/python3" ]] &&
       "$VENV_DIR/bin/python3" -m pip --version &>/dev/null; then
        log "Venv created: $("$VENV_DIR/bin/python3" --version)"
        return 0
    fi

    backup_incomplete_venv
    die "virtualenv did not produce a healthy environment at $VENV_DIR."
}

install_python_packages() {
    verify_local_cuda || die \
        "Exact local CUDA ${CUDA_VER%.*} toolkit is required at $CUDA_DIR before installing packages."
    set_cuda_env

    info "Upgrading Python packaging/build tools."
    pip_run install --upgrade pip setuptools wheel packaging psutil ninja
    "$VENV_DIR/bin/ninja" --version >/dev/null

    # Install torch first. flash-attn's setup imports torch and compares
    # torch.version.cuda with CUDA_HOME/nvcc, so order and matching matter.
    if torch_matches_profile; then
        info "PyTorch $TORCH_VER already matches $TORCH_FLAVOR."
    else
        info "Installing PyTorch $TORCH_VER from the $TORCH_FLAVOR index."
        pip_run install --upgrade --force-reinstall \
            "torch==${TORCH_VER}" \
            --index-url "$TORCH_IDX"
    fi
    PATH="$VENV_DIR/bin:$PATH" "$VENV_DIR/bin/python3" -c \
        'from torch.utils.cpp_extension import verify_ninja_availability; verify_ninja_availability(); print("[verify] PyTorch BuildExtension found ninja")'

    info "Installing LoomFormer runtime, data and model-conversion dependencies."
    pip_run install --upgrade \
        numpy \
        pyyaml \
        tokenizers \
        pyarrow \
        safetensors \
        transformers \
        jinja2 \
        einops \
        "huggingface_hub[cli]"

    verify_torch_toolchain

    if (( INSTALL_FLASH_ATTN )); then
        info "Installing FlashAttention for Ampere-or-newer GPU(s)."
        info "FlashAttention build: arches=$FLASH_ATTN_CUDA_ARCHS, MAX_JOBS=$BUILD_JOBS, NVCC_THREADS=$NVCC_BUILD_THREADS."
        local build_log
        local -a pipeline_status
        build_log=$(mktemp "${TMPDIR:-/tmp}/loomformer-flash-attn.XXXXXX.log")

        # pip only exposes Ninja's [current/total] events in verbose mode.
        # Keep that stream in a diagnostic log and render a compact live
        # counter instead of printing compiler command lines.
        set +e
        PATH="$VENV_DIR/bin:$PATH" \
            MAX_JOBS="$BUILD_JOBS" NVCC_THREADS="$NVCC_BUILD_THREADS" \
            FLASH_ATTN_CUDA_ARCHS="$FLASH_ATTN_CUDA_ARCHS" \
            "$VENV_DIR/bin/python3" -m pip install --verbose --upgrade \
                flash-attn --no-build-isolation 2>&1 |
            filter_flash_attn_build_output "$build_log"
        pipeline_status=("${PIPESTATUS[@]}")
        set -e

        if (( pipeline_status[0] != 0 || pipeline_status[1] != 0 )); then
            echo
            warn "FlashAttention build failed; last 80 log lines follow."
            tail -n 80 "$build_log" >&2
            die "Full FlashAttention build log retained at $build_log"
        fi

        safe_remove_build_log "$build_log"
        "$VENV_DIR/bin/python3" -c \
            'from flash_attn import flash_attn_varlen_func; print("[verify] flash_attn_varlen_func import OK")'
    else
        warn "No Ampere+ GPU detected: official FlashAttention-2 is unsupported here and was not installed. LoomFormer reference/SDPA attention remains available."
    fi

    pip_run check
    log "Python packages installed and dependency metadata is consistent."
}

torch_matches_profile() {
    local expected_cuda="${CUDA_VER%.*}"
    EXPECTED_TORCH_VERSION="$TORCH_VER" EXPECTED_TORCH_CUDA="$expected_cuda" \
        "$VENV_DIR/bin/python3" - <<'PY' >/dev/null 2>&1
import os
import torch

version = torch.__version__.split("+", 1)[0]
cuda = torch.version.cuda or ""
expected_version = os.environ["EXPECTED_TORCH_VERSION"]
expected_cuda = os.environ["EXPECTED_TORCH_CUDA"]
raise SystemExit(
    0 if version == expected_version and
    (cuda == expected_cuda or cuda.startswith(expected_cuda + "."))
    else 1
)
PY
}

verify_torch_toolchain() {
    local expected_cuda="${CUDA_VER%.*}"
    EXPECTED_TORCH_CUDA="$expected_cuda" "$VENV_DIR/bin/python3" - <<'PY'
import os
import subprocess
import torch

expected = os.environ["EXPECTED_TORCH_CUDA"]
torch_cuda = torch.version.cuda or ""
if not torch_cuda.startswith(expected + ".") and torch_cuda != expected:
    raise SystemExit(
        f"PyTorch/toolkit mismatch: torch={torch.__version__} reports CUDA "
        f"{torch_cuda!r}, expected {expected}.x"
    )
nvcc = subprocess.check_output(
    [os.path.join(os.environ["CUDA_HOME"], "bin", "nvcc"), "--version"],
    text=True,
)
print(f"[verify] torch={torch.__version__} torch.cuda={torch_cuda}")
print("[verify] " + next(line.strip() for line in nvcc.splitlines() if "release" in line))
if torch.cuda.is_available():
    for index in range(torch.cuda.device_count()):
        print(
            f"[verify] cuda:{index} {torch.cuda.get_device_name(index)} "
            f"SM{torch.cuda.get_device_capability(index)[0]}."
            f"{torch.cuda.get_device_capability(index)[1]}"
        )
PY
}

verify_loomformer() {
    step "Validating LoomFormer installation matrix"
    set_cuda_env
    (
        cd "$REPO_DIR"
        "$VENV_DIR/bin/python3" tests/run_matrix.py \
            --setup \
            --report "$VALIDATION_REPORT"
    )
    log "Complete synthetic PT/SFT/backend/DDP installation matrix passed."
}

print_attention_summary() {
    [[ -f "$VALIDATION_REPORT" ]] || return 0
    local line
    while IFS= read -r line; do
        [[ -n "$line" ]] || continue
        echo -e "  Attention    : ${CYAN}${line}${RESET}"
    done < <(
        "$VENV_DIR/bin/python3" - "$VALIDATION_REPORT" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    report = json.load(handle)
for item in report.get("attention", []):
    def backend(name, label):
        value = item.get(name, {})
        if not value.get("forward_backward", False):
            return f"{label}=unavailable"
        fused = value.get("fused_value")
        suffix = "" if fused is None else f", fused-value={'yes' if fused else 'no'}"
        return f"{label}=fwd+bwd{suffix}"

    fields = [
        backend("flash_attn", "flash-attn varlen"),
        backend("transformer_engine", "transformer-engine varlen"),
        backend("sdpa", "SDPA"),
        f"selected={item.get('selected', 'unknown')}",
    ]
    print(f"cuda:{item['index']} " + " · ".join(fields))
PY
    )
}

# ═══════════════════════════════════════════════════════════════════════════
#  Persistent environment
# ═══════════════════════════════════════════════════════════════════════════
write_env_file() {
    mkdir -p "$INSTALL_DIR"
    cat > "$ENV_FILE" <<EOF
# Generated by LoomFormer setup.sh. Re-running setup updates this file.
export CUDA_HOME="$CUDA_DIR"
export PATH="$CUDA_DIR/bin:\$PATH"
export LD_LIBRARY_PATH="$CUDA_DIR/lib64:\${LD_LIBRARY_PATH:-}"
export TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST"
export FLASH_ATTN_CUDA_ARCHS="$FLASH_ATTN_CUDA_ARCHS"
export MAX_JOBS="$BUILD_JOBS"
export NVCC_THREADS="\${NVCC_THREADS:-$NVCC_BUILD_THREADS}"
EOF
    chmod 0644 "$ENV_FILE"
    log "Wrote CUDA/build environment to $ENV_FILE."
}

patch_venv_activate() {
    local activate="$VENV_DIR/bin/activate"
    local marker="# >>> LoomFormer env <<<"
    [[ -f "$activate" ]] || return 0
    if grep -qF "$marker" "$activate"; then
        return 0
    fi
    cat >> "$activate" <<EOF

$marker
source "$ENV_FILE"
# <<< LoomFormer env <<<
EOF
    log "Patched venv activation hook."
}

patch_shell_rc() {
    local marker="# >>> LoomFormer env <<<"
    local rc
    local snippet
    snippet=$(cat <<EOF

$marker
# Added by LoomFormer setup.sh; the generated file is updated on reinstall.
[[ -f "$ENV_FILE" ]] && source "$ENV_FILE"
# <<< LoomFormer env <<<
EOF
)
    for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.profile"; do
        [[ -f "$rc" ]] || continue
        if grep -qF "$marker" "$rc"; then
            continue
        else
            printf '%s\n' "$snippet" >> "$rc"
            log "Patched $rc."
        fi
    done
}

# ═══════════════════════════════════════════════════════════════════════════
#  Top-level operations
# ═══════════════════════════════════════════════════════════════════════════
do_full_install() {
    mkdir -p "$INSTALL_DIR"

    step "Fetching LoomFormer"
    fetch_repo

    step "Installing CUDA $CUDA_VER toolkit"
    install_cuda_runfile

    step "Creating Python environment"
    install_venv

    step "Installing Python packages"
    install_python_packages

    write_env_file
    patch_venv_activate
    patch_shell_rc
    verify_loomformer
    print_summary
}

do_packages_only() {
    repo_exists || die "Repository not found at $REPO_DIR. Run full install first."
    verify_local_cuda || die \
        "CUDA ${CUDA_VER%.*} toolkit not found at $CUDA_DIR. Run full install first."
    install_venv
    install_python_packages
    write_env_file
    patch_venv_activate
    verify_loomformer
    print_summary
}

print_summary() {
    step "Setup complete"
    echo
    echo -e "${BOLD}${GREEN}  LoomFormer-Paraplex is ready.${RESET}"
    echo
    echo -e "  Repository   : ${CYAN}$REPO_DIR${RESET}"
    echo -e "  CUDA toolkit : ${CYAN}$CUDA_DIR${RESET}"
    echo -e "  PyTorch      : ${CYAN}$TORCH_VER ($TORCH_FLAVOR)${RESET}"
    echo -e "  Python venv  : ${CYAN}$VENV_DIR${RESET}"
    echo -e "  CUDA arches  : ${CYAN}$TORCH_CUDA_ARCH_LIST${RESET}"
    if (( INSTALL_FLASH_ATTN )); then
        echo -e "  FlashAttention: ${CYAN}arches=$FLASH_ATTN_CUDA_ARCHS, MAX_JOBS=$BUILD_JOBS, NVCC_THREADS=$NVCC_BUILD_THREADS${RESET}"
    fi
    print_attention_summary
    echo
    echo -e "  Activate:"
    echo -e "    ${BOLD}source $VENV_DIR/bin/activate${RESET}"
    echo -e "  Update repository:"
    echo -e "    ${BOLD}$REPO_DIR/setup.sh${RESET}  ${DIM}(choose option 3)${RESET}"
    echo -e "  Download FSSFT1:"
    echo -e "    ${BOLD}hf download srs6901/FSSFT1 --repo-type dataset --local-dir $REPO_DIR/datasets/sft/FSSFT1${RESET}"
    echo
    warn "Restart the shell or source $ENV_FILE to apply CUDA_HOME immediately."
}

main() {
    print_logo
    probe_tools
    if [[ ! -t 0 ]]; then
        warn "Non-interactive stdin — running full install."
        do_full_install
        return
    fi
    menu
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
