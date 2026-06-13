# XReflection External Integration

This directory is reserved for the external XReflection checkout and local
notes.  It is intentionally separate from the ERRNet code so RDNet can be used
as an external comparison method without changing the baseline implementation.

Suggested layout:

```bash
external_models/xreflection/
  README.md
  XReflection/              # cloned from https://github.com/hainuo-wang/XReflection
```

The project scripts do not import XReflection internals by default.  They run
the official XReflection command in a subprocess and normalize the saved clean
images into the ERRNet result layout:

```text
results/xreflection_rdnet/<dataset>/<image_stem>/xreflection_rdnet.png
```

Use `scripts/run_xreflection_rdnet.py --command_template` if the upstream
XReflection inference command changes.
