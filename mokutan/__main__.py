"""Usage: {NAME} [options] [HOST] [PORT]

Start the {DNAME} instance on http://localhost:2323 by default.
To make it accessible from outside this machine, use "0.0.0.0" as HOST.

Arguments:
    HOST  Adress to bind the server to, 127.0.0.1 if unspecified.
    PORT  Port to listen on, 2323 if unspecified.

Options:
    -r DIR, --reload DIR  Restart {DNAME} when source code files in DIR change.
    -h, --help            Show this help and exit.
    --version             Show the {DNAME} version and exit.
"""

import os
from pathlib import Path

import docopt
import uvicorn

from . import DISPLAY_NAME, NAME, __version__


def run() -> None:
    doc = (__doc__ or "").format(NAME=NAME, DNAME=DISPLAY_NAME)
    args = docopt.docopt(doc, version=__version__)
    rdir = args["--reload"]
    if rdir:
        rdir = Path(rdir).resolve()  # Might change cwd later, "." would break
        os.environ["UVICORN_RELOAD"] = str(rdir)

    uvicorn.run(
        f"{NAME}.app:app",
        host=args["HOST"] or "127.0.0.1",
        port=int(args["PORT"] or 2323),
        reload=bool(rdir),
        reload_dirs=[str(rdir)] if rdir else [],
        timeout_graceful_shutdown=0,
    )


if __name__ == "__main__":
    run()
