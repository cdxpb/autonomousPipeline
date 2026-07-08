import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--sync-destination', default='/code')
print("Default is:", parser.parse_args([]).sync_destination)
