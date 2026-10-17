"""
Custom icpx -fsycl build of the SYCL backend (module name exllamav3_sycl_ext).

The CUDA and ROCm builds go through torch's JIT; the SYCL build is hand-rolled because icpx
takes -fsycl and the oneAPI link line directly: the kernels compile to the target GPU's AOT
image (spir64_gen + the device target, e.g. dg2 for Arc) and the extension links torch XPU and
oneMKL BLAS in addition to torch itself.

Kernels run on the current torch XPU stream (c10::xpu::getCurrentXPUStream()) so they stay
ordered with torch ops.
"""

import importlib.machinery
import importlib.util
import os
import subprocess
import sys
import sysconfig
import torch
import torch.utils.cpp_extension

extension_name = "exllamav3_sycl_ext"
verbose = False

# Build cache (same idea as torch's JIT cache: objects and the .so land here and are reused
# while the source hashes match)
build_dir = os.path.join(
    torch.utils.cpp_extension.get_default_build_root(),
    extension_name,
)


def _find_oneapi() -> tuple[str, str]:
    """Returns (compiler bin dir, MKL include dir with oneapi/mkl headers or None). Probes
    torch's SYCL_HOME first (the same install torch XPU was built against), then the
    modulefiles/setvars layout."""
    candidates = []

    sycl_home = getattr(torch.utils.cpp_extension, "SYCL_HOME", None)
    if sycl_home:
        candidates.append(sycl_home)

    # Typical oneAPI roots, newest version first
    for root in ("/opt/intel/oneapi", os.path.expanduser("~/intel/oneapi")):
        if not os.path.isdir(root):
            continue
        if os.path.isdir(os.path.join(root, "compiler", "latest")):
            candidates.append(os.path.join(root, "compiler", "latest"))
        versions = sorted((v for v in os.listdir(os.path.join(root, "compiler")) if v[0].isdigit()),
                          reverse = True) if os.path.isdir(os.path.join(root, "compiler")) else []
        for v in versions:
            candidates.append(os.path.join(root, "compiler", v))
        mkls = sorted((v for v in os.listdir(os.path.join(root, "mkl")) if v[0].isdigit()),
                      reverse = True) if os.path.isdir(os.path.join(root, "mkl")) else []
        if mkls:
            candidates.append(os.path.join(root, "mkl", mkls[0]))

    compiler_bin = None
    mkl_include = None
    for c in candidates:
        if compiler_bin is None:
            for sub in ("bin",):
                if os.path.exists(os.path.join(c, sub, "icpx")):
                    compiler_bin = os.path.join(c, sub)
        if mkl_include is None and os.path.exists(os.path.join(c, "include", "oneapi", "mkl", "blas.hpp")):
            mkl_include = os.path.join(c, "include")
    if compiler_bin is None:
        raise RuntimeError(
            "icpx not found: source /opt/intel/oneapi/setvars.sh or install the Intel oneAPI "
            "DPC++ compiler to build the SYCL backend"
        )
    return compiler_bin, mkl_include


def _find_mkl_lib(sycl_home: str | None, mkl_include: str | None) -> str | None:
    """The MKL lib dir to link against. The pip wheels torch XPU ships with carry domain
    libs (libmkl_sycl_blas.so.N) built against the same libsycl the runtime uses — prefer
    those (co-located with torch's SYCL runtime) over the oneAPI toolkit's monolithic
    libmkl_sycl.so, which would drag in a second, mismatched libsycl at runtime."""
    if sycl_home and os.path.isdir(os.path.join(sycl_home, "lib")):
        lib = os.path.join(sycl_home, "lib")
        if any(f.startswith("libmkl_sycl_") for f in os.listdir(lib)):
            return lib
    if mkl_include:
        # OneAPI toolkit: mkl/<ver>/include -> ../lib (or the compiler root's sibling)
        root = os.path.dirname(mkl_include)
        if os.path.exists(os.path.join(root, "lib", "libmkl_sycl.so")):
            return os.path.join(root, "lib")
    for c in ("/opt/intel/oneapi/mkl/latest/lib", os.path.expanduser("~/intel/oneapi/mkl/latest/lib")):
        if os.path.exists(os.path.join(c, "libmkl_sycl.so")):
            return c
    return None


def _mkl_link_flags(mkl_lib: str | None) -> list[str]:
    """Link flags for the MKL dir found by _find_mkl_lib: the pip domain lib by exact
    filename (-l:<file>) since there is no libmkl_sycl_blas.so symlink, the toolkit's
    monolithic lib with -lmkl_sycl."""
    if not mkl_lib:
        return []
    domain = sorted(
        f for f in os.listdir(mkl_lib)
        if f.startswith("libmkl_sycl_blas.so") and not f.endswith(".so.0") and ".so." in f
    )
    # Prefer the plain .so.<N> the package actually ships
    domain = [f for f in domain if f.count(".so.") == 1] or domain
    if domain:
        return [f"-l:{domain[0]}"]
    return ["-lmkl_sycl"]


def _sycl_target_flags() -> list[str]:
    """AOT target for the visible GPU(s) when the ocloc offline compiler is available; otherwise
    JIT-able spir64. torch.xpu.get_arch_list() reports every target the *runtime* supports
    (dg2 = Arc A-series, bmg = Arc B-series, pvc = datacenter, arl-h/mtl-h/lnl-m/ptl-* = client
    iGPUs) — not the visible device's own arch — so iGPUs are matched by name: AOT images built
    for dg2/bmg/pvc do not load on iGPU silicon. The SYCL_TARGET_DEVICE env var overrides
    detection with an exact ocloc device name."""
    try:
        archs = torch.xpu.get_arch_list()
    except Exception:
        archs = []

    def _igpu_target(n: str) -> str | None:
        # iGPU name -> ocloc device target; checked against the runtime's arch list so an
        # unrecognized iGPU falls back to JIT (spir64) rather than a mismatched AOT image
        if "mtl" in n or "meteor" in n or "130t" in n: return "mtl-h"
        if "lnl" in n or "lunar" in n: return "lnl-m"
        if "140t" in n or "110t" in n or "arrow" in n or "arl" in n: return "arl-h"
        if "ptl" in n or "panther" in n: return "ptl-h"
        return next((t for t in ("arl-h", "mtl-h", "lnl-m", "ptl-h", "ptl-u") if t in archs), None)

    # One target per visible device, deduplicated
    targets = set()
    forced = os.environ.get("SYCL_TARGET_DEVICE", "").strip().lower()
    if forced:
        targets.add(forced)
    else:
        try:
            for dev in range(torch.xpu.device_count()):
                name = torch.xpu.get_device_name(dev)
                n = name.lower()
                if " b" in n and any(f"b{d}" in n for d in range(5, 8)):
                    targets.add("bmg")
                elif "140t" in n or "110t" in n or "iris" in n or "uhd" in n:
                    t = _igpu_target(n)
                    if t: targets.add(t)
                elif "arc" in n and not any(m in n for m in ("a3", "a5", "a7", "pro")):
                    # bare "Arc(...) Graphics"-style names without a dGPU model number = iGPU
                    t = _igpu_target(n)
                    if t: targets.add(t)
                elif any(a in n for a in ("arc", "a770", "a750", "a380")):
                    targets.add("dg2")
                elif any(a in n for a in ("max", "pvc", "9460")):
                    targets.add("pvc")
        except Exception:
            pass
        if not targets:
            if "dg2" in archs: targets.add("dg2")
            elif "bmg" in archs: targets.add("bmg")
            elif "pvc" in archs: targets.add("pvc")
    # ocloc (the offline compiler) is required for AOT images; without it the kernels keep the
    # JIT-able spir64 target, which the runtime compiles at first launch
    have_ocloc = any(
        os.path.exists(os.path.join(d, "ocloc"))
        for d in os.environ.get("PATH", "").split(os.pathsep)
    )
    if not targets or not have_ocloc:
        return ["-fsycl-targets=spir64"]
    # -fsycl-targets must be a SYCL target (spir64_gen = AOT, spir64 = JIT); the device
    # (dg2/bmg/pvc, an ocloc device name) goes to the backend via -Xsycl-target-backend
    flags = ["-fsycl-targets=spir64_gen",
             "-Xsycl-target-backend=spir64_gen",
             "-device " + ",".join(sorted(targets))]
    return flags


def _source_hash(sources: list[str], cflags: list[str], ldflags: list[str],
                 compiler_version: str = "") -> str:
    import hashlib
    h = hashlib.sha256()
    for s in sources:
        h.update(s.encode())
        try:
            with open(s, "rb") as f:
                h.update(f.read())
        except OSError:
            pass
    for f in cflags + ldflags:
        h.update(f.encode())
    h.update(torch.__version__.encode())
    # Binaries are only compatible with the SYCL runtime they were compiled against —
    # an .so built by a different icpx must rebuild (mixing runtimes segfaults in
    # __sycl_register_lib at load time)
    h.update(compiler_version.encode())
    return h.hexdigest()[:16]


def build_sycl_extension(verbosity: bool = False, ext_debug: bool = False):
    global verbose
    verbose = verbosity

    compiler_bin, mkl_include = _find_oneapi()
    sycl_home = getattr(torch.utils.cpp_extension, "SYCL_HOME", None)
    mkl_lib = _find_mkl_lib(sycl_home, mkl_include)
    icpx = os.path.join(compiler_bin, "icpx")
    ver = subprocess.run([icpx, "--version"], capture_output = True, text = True)
    icpx_version = (ver.stdout + ver.stderr).strip() if ver.returncode == 0 else ""

    library_dir = os.path.dirname(os.path.abspath(__file__))
    package_dir = os.path.dirname(library_dir)
    sources_dir = os.path.join(package_dir, "exllamav3_ext", "sycl")
    sources_dir = os.path.normpath(sources_dir)
    if not os.path.isdir(sources_dir):
        raise RuntimeError(f"SYCL sources not found at {sources_dir}")

    sources = sorted(
        os.path.join(sources_dir, f) for f in os.listdir(sources_dir) if f.endswith(".cpp")
    )
    if not sources:
        raise RuntimeError(f"No .cpp sources in {sources_dir}")

    torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
    torch_include = os.path.join(os.path.dirname(torch.__file__), "include")
    python_include = sysconfig.get_paths()["include"]

    sycl_include = os.path.join(sycl_home, "include") if sycl_home else None

    include_flags = []
    for inc in (torch_include,
                os.path.join(torch_include, "torch", "csrc", "api", "include"),
                sycl_include,
                mkl_include,
                python_include,
                sources_dir):
        if inc and os.path.isdir(inc):
            include_flags += ["-I", inc]

    cflags = [
        "-std=c++20",
        "-fsycl",
        "-fPIC",
        "-O3" if not ext_debug else "-O0",
        "-DTORCH_EXTENSION_NAME=" + extension_name,
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
    ] + include_flags + _sycl_target_flags()
    if ext_debug:
        cflags += ["-g"]

    ldflags = [
        # -fsycl on the link line performs the SYCL device-link step: without it the device
        # images are never packaged into the .so and every kernel launch segfaults
        "-fsycl",
        "-shared",
        f"-L{torch_lib}",
        "-ltorch", "-ltorch_cpu", "-ltorch_python", "-lc10", "-lc10_xpu", "-ltorch_xpu",
    ]
    if sycl_home:
        ldflags += [f"-L{os.path.join(sycl_home, 'lib')}", "-lsycl"]
    if mkl_lib:
        ldflags += [f"-L{mkl_lib}"] + _mkl_link_flags(mkl_lib)
    # Rpaths so the .so resolves torch/oneAPI deps at runtime regardless of LD_LIBRARY_PATH
    # (the loader only reads that variable at process startup, so setting it here — as the
    # code below does — never helps the current process)
    for rp in (torch_lib,
               os.path.join(sycl_home, "lib") if sycl_home else None,
               mkl_lib):
        if rp:
            ldflags += [f"-Wl,-rpath,{rp}"]
    # AOT targets for the visible GPU(s) apply at device-link time
    ldflags += _sycl_target_flags()

    out_name = extension_name + ".so"
    out_path = os.path.join(build_dir, out_name)

    # Rebuild when any source or flag changes
    h = _source_hash(sources, cflags, ldflags, icpx_version)
    marker = os.path.join(build_dir, f".hash-{h}")
    if os.path.exists(marker) and os.path.exists(out_path):
        if verbose:
            print(f" -- SYCL extension cache hit: {out_path}")
    else:
        os.makedirs(build_dir, exist_ok=True)
        objects = []
        for src in sources:
            obj = os.path.join(build_dir, os.path.basename(src) + ".o")
            cmd = [icpx] + cflags + ["-c", src, "-o", obj]
            if verbose:
                print(" --", " ".join(cmd))
            r = subprocess.run(cmd, capture_output = True, text = True)
            if r.returncode != 0:
                print(r.stdout)
                print(r.stderr, file = sys.stderr)
                raise RuntimeError(f"SYCL compile failed: {src}\n{r.stderr[-4000:]}")
            elif verbose and r.stderr.strip():
                print(r.stderr)
            objects.append(obj)

        link_cmd = [icpx] + ldflags + objects + ["-o", out_path]
        if verbose:
            print(" --", " ".join(link_cmd))
        r = subprocess.run(link_cmd, capture_output = True, text = True)
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr, file = sys.stderr)
            raise RuntimeError(f"SYCL link failed:\n{r.stderr[-4000:]}")
        with open(marker, "w") as f:
            f.write(h)
        # Prune stale objects from previous hashes
        for f in os.listdir(build_dir):
            if f.endswith(".o") and not os.path.exists(os.path.join(build_dir, f)):
                pass

    # Runtime library paths: torch lib (usually already on the loader path via torch itself) and
    # the oneAPI runtime libs
    for p in (torch_lib, os.path.join(sycl_home, "lib") if sycl_home else None, mkl_lib):
        if p and os.path.isdir(p) and p not in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
            os.environ["LD_LIBRARY_PATH"] = p + ":" + os.environ.get("LD_LIBRARY_PATH", "")

    spec = importlib.util.spec_from_file_location(extension_name, out_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[extension_name] = mod
    return mod
