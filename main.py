from utils.utils import read_options,setup_seed
#export HF_ENDPOINT="https://hf-mirror.com"
def run_fed(args):
    if args.fed_alg == "fedit":
        from fed.FedIT import FedIT
        fed = FedIT(args)
        fed.run()
    elif args.fed_alg == "ffalora":
        from fed.FFALora import FFALora
        fed = FFALora(args)
        fed.run()
    elif args.fed_alg == "flora":
        from fed.FLora import FLora
        fed = FLora(args)
        fed.run()
    elif args.fed_alg == "fedls":
        from fed.FedLS import FedLS
        fed = FedLS(args)
        fed.run()
    elif args.fed_alg == "fedex":
        from fed.FedExLora import FedEx
        fed = FedEx(args)
        fed.run()
    elif args.fed_alg == "fedsvd":
        from fed.FedSVD import FedSVD
        fed = FedSVD(args)
        fed.run()
    else:
        raise ValueError(f"Unknown federated learning algorithm: {args.fed_alg}")

if __name__ == "__main__":
    args = read_options()
    setup_seed(args.seed)
    run_fed(args)

    
