import yaml

def read_config(path):
    # read yaml and return contents 
    with open(path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Error read yaml file {path}: {e}")