from fastie.data.uie.doccano import convert_uie_data, parse_doccano_args

if __name__ == "__main__":
    args = parse_doccano_args()
    convert_uie_data(args)
