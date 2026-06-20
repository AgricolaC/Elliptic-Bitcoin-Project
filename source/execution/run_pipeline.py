import argparse
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.dirname(HERE)
if SOURCE not in sys.path:
    sys.path.insert(0, SOURCE)

def main():
    parser = argparse.ArgumentParser(description="Unified Execution Engine for Elliptic Pipeline")
    parser.add_argument("--phase", type=str, required=True, 
                        choices=["f1", "f2", "f3", "f4"],
                        help="The phase to execute")
    # Pass arbitrary args to the phase script if needed by mocking sys.argv
    args, unknown = parser.parse_known_args()
    
    sys.argv = [sys.argv[0]] + unknown
    
    if args.phase == "f1":
        from phases import f1_walk_forward
        f1_walk_forward.run()
    elif args.phase == "f2":
        from phases import f2_lstm
        f2_lstm.run()
    elif args.phase == "f3":
        from phases import f3_baselines
        f3_baselines.run()
    elif args.phase == "f4":
        from phases import f4_exponential_decay
        f4_exponential_decay.run()

if __name__ == "__main__":
    main()
