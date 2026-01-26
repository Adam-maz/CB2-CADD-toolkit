import pandas as pd
import os
import argparse

import torch
from torch_geometric.utils import from_smiles
from torch_geometric.loader import DataLoader
from torch_geometric.nn.models import AttentiveFP


class Predictor:
    
    def __init__(self, text):
        self.text = text
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = AttentiveFP(in_channels=9, hidden_channels=96, out_channels=1, edge_dim=3, num_layers=6, num_timesteps=3, dropout=0.042543)
        self.state_dict = torch.load("attentivefp_model_final_full_data.pth", map_location=self.device)
    
    def load_model(self):
        self.model.load_state_dict(self.state_dict)
        self.model.to(self.device)
        self.model.eval()

    def load_data(self):
        if self.text.endswith('.csv'):
            self.df = pd.read_csv(self.text, sep=';')
        else:
            self.df = pd.DataFrame({'smiles': [self.text]})
        return self.df

    def make_graphs(self):
        self.graphs_list = []
        for sm in self.df["smiles"]:
            g = from_smiles(sm)
            g.x = g.x.float()
            self.graphs_list.append(g)
        return self.graphs_list

    def dataloader(self):
        self.loader = DataLoader(self.graphs_list, batch_size=64, shuffle=False)
        return self.loader

    @torch.no_grad()
    def predict_affinity(self):
        preds = []
        self.model.eval()
        for data in self.loader:
            data = data.to(self.device)
            out = self.model(data.x, data.edge_index, data.edge_attr, data.batch)
            preds.append(out.cpu())
    
        predictions = torch.cat(preds).numpy()
        self.df['Predicted_pKi'] = predictions
        return self.df[['smiles', 'Predicted_pKi']]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='Provide string which corresponds to one SMILES or PATH to .csv file with multiple SMILES')
    args = parser.parse_args()

    predictor = Predictor(args.path)
    predictor.load_model()
    predictor.load_data()
    predictor.make_graphs()
    predictor.dataloader()
    output = predictor.predict_affinity()
    output = output.sort_values(by='Predicted_pKi', ascending=False) 
    print('-----------------------------------')
    print('Predicted Affinity: ')
    print(output)
    print('-----------------------------------')

    path_for_store_output = os.path.join(os.path.dirname(os.getcwd()), 'CB2R_predictions.csv')
    save = str(input(f'Do you want to save predictions at {path_for_store_output}? y/n: '))

    if save == 'y':
        output.to_csv(path_for_store_output, index=False)
        print('Results saved')
    else:
        print('Program terminated.')
        

if __name__ == "__main__":
    main()   

        