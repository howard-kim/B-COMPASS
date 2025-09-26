from __future__ import annotations

from experiment import ExperimentConfig
from cli import parse_args
from plot import plot_results
from experiment import CompassRepro


def main():
    args = parse_args()
    cfg = ExperimentConfig(
        circuits=args.circuits,
        depths=args.depths,
        combos=args.combos,
        shots=args.shots,
        backend=args.backend,
    )
    runner = CompassRepro(cfg)
    df = runner.run()
    if args.save:
        df.to_csv(args.save, index=False)
        print(f"Saved: {args.save}")
    if args.plot:
        plot_results(df)
    else:
        print(df.head())


if __name__ == "__main__":
    main()
