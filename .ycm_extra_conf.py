#!/usr/bin/env python3
# ---------------------------------------------------------
#   Name: .ycm_extra_conf.py
#   Author: xyy15926
#   Created: 2025-09-22 20:11:41
#   Updated: 2025-12-11 22:22:30
#   Description:
# ---------------------------------------------------------

def Settings(**kwargs):
    from pathlib import Path
    return {
        "interpreter_path": str(
            Path(__file__).parent
            / ".pixi/envs/default/bin/python3"
        ),
    }
