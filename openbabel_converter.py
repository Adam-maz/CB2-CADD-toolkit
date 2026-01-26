from openbabel import pybel
import pandas as pd
import argparse
import os



class MolTypeConverter:
    

    def __init__(self, path):
        self.path = path
        self.out_path = os.path.join(os.path.dirname(self.path), 'converted_ligands')
        os.makedirs(self.out_path, exist_ok=True)

    def data_loading(self):
        self.df = pd.read_csv(self.path, sep=';')
        self.df.columns = self.df.columns.str.strip().str.lower()

    def convert(self):
        for sm, idx in zip(self.df['smiles'], self.df['id']):
            mol = pybel.readstring("smi", sm)
            mol.addh()
            mol.make3D()
            mol.write("pdb", os.path.join(self.out_path, f'{idx}.pdb'))
	


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Provide path string to the directory which contains desired molecules')
    args = parser.parse_args()

    converter = MolTypeConverter(args.path)
    converter.data_loading()
    converter.convert()


if __name__ == "__main__":
    main()