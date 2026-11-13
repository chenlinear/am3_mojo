"""PEP 517 build backend of am3_mojo (in-tree, see pyproject.toml).

`pip install .` and `uv pip install .` call it. It compiles
src/am3_mojo_module.mojo into the extension module `am3_mojo` with the Mojo
compiler of the build environment (the `mojo-compiler` package from PyPI).

The module needs Mojo's runtime libraries (libKGENCompilerRTShared and the two
that it loads), which `mojo-compiler` installs into <site-packages>/modular/lib.
So the wheel depends on `mojo-compiler`, and the module finds the libraries
through an rpath relative to itself, not in the build environment. The Mojo
driver also links in an absolute rpath into the (temporary) build environment.
This rpath is searched first, and install_name_tool (macOS) or patchelf (Linux,
from PyPI into the build environment) removes it.

The module is compiled for the building machine's CPU (Mojo's default), so the
wheel is meant for that machine only. Mojo links with the C compiler on the PATH
(Xcode Command Line Tools on macOS, gcc or clang on Linux).
"""

import base64, csv, gzip, hashlib, io, os, re, shutil, subprocess, sys
import sysconfig, tarfile, tempfile, tomllib, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODULE = "am3_mojo"


def _project():
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        return tomllib.load(f)["project"]


def _dist_name(p):
    return re.sub(r"[-_.]+", "_", p["name"])


def _metadata(p):
    lines = [
        "Metadata-Version: 2.1",
        f"Name: {p['name']}",
        f"Version: {p['version']}",
        f"Summary: {p['description']}",
        f"License: {p['license']['text']}",
        f"Requires-Python: {p['requires-python']}",
    ]
    lines += [f"Requires-Dist: {d}" for d in p.get("dependencies", [])]
    lines += ["Description-Content-Type: text/markdown", ""]
    with open(
        os.path.join(ROOT, p.get("readme", "README.md")), encoding="utf-8"
    ) as f:
        return "\n".join(lines) + "\n" + f.read()


def _mojo():
    """Command that runs the build environment's Mojo compiler
    (mojo-compiler==1.2.0), whatever Mojo is on the PATH. It calls the package's
    own `mojo` launcher through this interpreter, because pip keeps the build
    environment's scripts in a directory that is not on sys.prefix."""
    try:
        import mojo._entrypoints
    except ImportError:
        raise RuntimeError(
            "Mojo compiler not found: install the `mojo-compiler` package"
        )
    return [
        sys.executable,
        "-c",
        "import sys; from mojo._entrypoints import exec_mojo; sys.argv[0] "
        '= "mojo"; exec_mojo()',
    ]


def _platform_tag(so_path):
    plat = sysconfig.get_platform().replace("-", "_").replace(".", "_")
    if sys.platform == "darwin":
        # Read the deployment target that the module was linked for.
        # From macOS 11 on, a tag names the major version only (macosx_15_0
        # for 15.4), as pip and the `wheel` package read it.
        out = subprocess.run(
            ["otool", "-l", so_path], capture_output=True, text=True
        ).stdout
        m = re.search(r"minos (\d+)\.(\d+)", out)
        if m:
            major, minor = int(m.group(1)), int(m.group(2))
            if major >= 11:
                minor = 0
            plat = f"macosx_{major}_{minor}_{plat.rsplit('_', 1)[-1]}"
    elif sys.platform.startswith("linux"):
        plat = plat.replace("linux", "manylinux_2_28")
    return plat


def _compile(out_dir):
    # The bindings are generated from tools/ (`make bindings`), not here.
    # Without this check, a module generated from other API descriptions or
    # another generator is built as it is.
    if subprocess.run(
        [
            sys.executable,
            os.path.join(ROOT, "tools", "gen_bindings.py"),
            "--check",
        ]
    ).returncode:
        raise RuntimeError(
            "src/am3_mojo_module.mojo is not generated from the current"
            " tools/gen_bindings.py and tools/*_api.json: run `make bindings`"
        )
    ext = sysconfig.get_config_var("EXT_SUFFIX")
    so = os.path.join(out_dir, MODULE + ext)
    rpath = (
        "@loader_path/modular/lib"
        if sys.platform == "darwin"
        else "$ORIGIN/modular/lib"
    )
    cmd = _mojo() + [
        "build",
        "-D",
        "ASSERT=none",
        "--emit",
        "shared-lib",
        "-I",
        os.path.join(ROOT, "src"),
        os.path.join(ROOT, "src", "am3_mojo_module.mojo"),
        "-o",
        so,
        "-Xlinker",
        "-rpath",
        "-Xlinker",
        rpath,
    ]
    print("compiling am3_mojo:", " ".join(cmd), file=sys.stderr)
    subprocess.run(cmd, check=True)
    if sys.platform == "darwin":
        out = subprocess.run(
            ["otool", "-l", so], capture_output=True, text=True, check=True
        ).stdout
        for p in re.findall(r"path (/\S+) \(offset", out):
            subprocess.run(
                ["install_name_tool", "-delete_rpath", p, so], check=True
            )
        subprocess.run(
            ["install_name_tool", "-id", os.path.basename(so), so], check=True
        )
        subprocess.run(
            ["codesign", "--force", "--sign", "-", so],
            check=True,
            capture_output=True,
        )
    elif sys.platform.startswith("linux"):
        # pip and uv put the build environment's patchelf on the PATH
        patchelf = shutil.which("patchelf")
        if patchelf is None:
            raise RuntimeError("patchelf not found: install `patchelf`")
        subprocess.run([patchelf, "--set-rpath", rpath, so], check=True)
    return so


def _record_line(arcname, data):
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(data).digest())
        .rstrip(b"=")
        .decode()
    )
    return [arcname, f"sha256={digest}", str(len(data))]


def _write_wheel(wheel_directory, files, tag, p):
    name = f"{_dist_name(p)}-{p['version']}-{tag}.whl"
    dist_info = f"{_dist_name(p)}-{p['version']}.dist-info"
    files = dict(files)
    files[f"{dist_info}/METADATA"] = _metadata(p).encode()
    files[f"{dist_info}/WHEEL"] = (
        "Wheel-Version: 1.0\nGenerator: am3_mojo mojo_build_backend\n"
        f"Root-Is-Purelib: false\nTag: {tag}\n"
    ).encode()
    with open(os.path.join(ROOT, "LICENSE"), "rb") as f:
        files[f"{dist_info}/LICENSE"] = f.read()
    rows = [_record_line(a, d) for a, d in files.items()] + [
        [f"{dist_info}/RECORD", "", ""]
    ]
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    files[f"{dist_info}/RECORD"] = buf.getvalue().encode()
    path = os.path.join(wheel_directory, name)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for arcname, data in files.items():
            info = zipfile.ZipInfo(arcname, date_time=(2026, 1, 1, 0, 0, 0))
            info.external_attr = (
                0o755 if arcname.endswith(".so") else 0o644
            ) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, data)
    return name


# ---------------------------------------------------------------- PEP 517 hooks
def get_requires_for_build_wheel(config_settings=None):
    return ["patchelf>=0.17"] if sys.platform.startswith("linux") else []


def get_requires_for_build_sdist(config_settings=None):
    return []


def prepare_metadata_for_build_wheel(metadata_directory, config_settings=None):
    p = _project()
    dist_info = f"{_dist_name(p)}-{p['version']}.dist-info"
    os.makedirs(os.path.join(metadata_directory, dist_info), exist_ok=True)
    with open(
        os.path.join(metadata_directory, dist_info, "METADATA"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(_metadata(p))
    return dist_info


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    p = _project()
    with tempfile.TemporaryDirectory() as tmp:
        so = _compile(tmp)
        with open(so, "rb") as f:
            data = f.read()
        py = f"cp{sys.version_info.major}{sys.version_info.minor}"
        tag = f"{py}-{py}-{_platform_tag(so)}"
        return _write_wheel(
            wheel_directory, {os.path.basename(so): data}, tag, p
        )


def build_sdist(sdist_directory, config_settings=None):
    p = _project()
    base = f"{_dist_name(p)}-{p['version']}"
    path = os.path.join(sdist_directory, base + ".tar.gz")
    # (of tools/, what a build needs: this backend and the bindings' inputs)
    include = [
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "src",
        "tools/am3_api.json",
        "tools/am3_mojo_api.json",
        "tools/gen_bindings.py",
        "tools/mojo_build_backend.py",
    ]
    skip = ("__pycache__", ".DS_Store")
    # the same archive from the same sources: one time for every entry
    # (SOURCE_DATE_EPOCH, else 1 Jan 1980)
    epoch = int(os.environ.get("SOURCE_DATE_EPOCH", 315532800))

    def keep(ti):
        if any(s in ti.name.split("/") for s in skip):
            return None
        ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = epoch, 0, 0, "", ""
        return ti

    with (
        gzip.GzipFile(path, "wb", mtime=0) as z,
        tarfile.open(fileobj=z, mode="w") as t,
    ):
        for item in include:
            t.add(
                os.path.join(ROOT, item), arcname=f"{base}/{item}", filter=keep
            )
        info = tarfile.TarInfo(f"{base}/PKG-INFO")
        data = _metadata(p).encode()
        info.size, info.mtime = len(data), epoch
        t.addfile(info, io.BytesIO(data))
    return os.path.basename(path)
