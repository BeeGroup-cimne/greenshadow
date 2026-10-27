# -*- coding: utf-8 -*-
"""``python -m greenshadow.pipelines <region> ...``"""

import sys

from . import catalonia

REGIONS = {
    "catalonia": catalonia.main,
}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in REGIONS:
        sys.stderr.write(f"usage: python -m greenshadow.pipelines {{{','.join(REGIONS)}}} ...\n")
        return 2
    REGIONS[argv[0]](argv[1:], prog=f"python -m greenshadow.pipelines {argv[0]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
