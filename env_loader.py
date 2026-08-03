"""Load key=value pairs from the repo's .env into os.environ (no external deps).

Importing this module (``import env_loader``) populates os.environ from the
nearest .env found by walking up from this file. Existing environment variables
are NOT overwritten, so you can still override any key from the shell.
"""
import os
import pathlib


def load_env():
    here = pathlib.Path(__file__).resolve()
    for d in [here.parent, *here.parents]:
        env = d / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
            return str(env)
    return None


load_env()
