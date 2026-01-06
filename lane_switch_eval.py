import sys
import runpy
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "drivestudio"))

from ls_utils.eval_utils import (
    parse_lateral_offset_from_cli,
    install_fixed_offset_trajectory,
    install_render_traj_key_renamer,
    install_post_resume_hotfix
)

LATERAL = parse_lateral_offset_from_cli(sys.argv[1:])
install_fixed_offset_trajectory(LATERAL)
install_render_traj_key_renamer(LATERAL)
install_post_resume_hotfix()



if __name__ == "__main__":
    runpy.run_path("drivestudio/tools/eval.py", run_name="__main__")