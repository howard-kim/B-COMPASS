
import argparse
# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="COMPASS reproduction scaffold")
    p.add_argument("--circuits", nargs="+", default=["bv", "qaoa"], help="bv qaoa pea")
    p.add_argument("--depths", nargs="+", type=int, default=[1, 2, 3], help="list of depths/p")
    p.add_argument("--combos", nargs="+", default=["na-sa-al", "sa-sa-al", "ol1", "ol3"], help="pass combos")
    p.add_argument("--shots", type=int, default=2000)
    p.add_argument("--backend", type=str, default="aer")
    p.add_argument("--save", type=str, default="")
    p.add_argument("--plot", action="store_true")
    return p.parse_args()