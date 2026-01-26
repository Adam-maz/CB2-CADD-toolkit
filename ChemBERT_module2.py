from transformers import RobertaForCausalLM, RobertaTokenizer, pipeline
import torch
from datasets import Dataset
import rdkit
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')


class ChemBERT:
    def __init__(
        self,
        smiles: str,
        iterations: int,
        max_new_tokens: int,
        do_sample: bool,
        num_beams: int,
        temperature: float,
        top_k: int,
        top_p: float,
        num_return_sequences: int,
        repetition_penalty: float):

        self.model = RobertaForCausalLM.from_pretrained("gokceuludogan/ChemBERTaLM", dtype=torch.float16)
        self.tokenizer = RobertaTokenizer.from_pretrained("gokceuludogan/ChemBERTaLM")

        self.smiles = smiles
        self.iterations = iterations
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.num_beams = num_beams
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.num_return_sequences = num_return_sequences
        self.repetition_penalty = repetition_penalty
    
    def config(self):
        self.dev = 0 if torch.cuda.is_available() else -1
        if self.dev == 0:
            self.model = self.model.to("cuda")

        self.model.eval()
        torch.set_grad_enabled(False)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer, device=self.dev)

    def generate(self, batch_size: int = 32):
        prompts = [self.smiles] * self.iterations
        dataset = Dataset.from_dict({"text": prompts})

        outputs = self.generator(
            list(dataset["text"]),
            batch_size=batch_size,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            num_beams=self.num_beams,
            num_return_sequences=self.num_return_sequences,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty
        )

        self.l = [o[0]["generated_text"] for o in outputs]

    def sanitize(self) -> tuple[list[rdkit.Chem.rdchem.Mol], list[str]]:
        mols_with_Nones = [Chem.MolFromSmiles(mol) for mol in self.l if mol is not None]
        mols_without_Nones = [mol for mol in mols_with_Nones if mol is not None]
        mols_with_smiles = [Chem.MolToSmiles(mol) for mol in mols_without_Nones]
        return mols_without_Nones, mols_with_smiles


def combinatorial_synthesis(
        smiles: str,
        iterations: int = 20,
        max_new_tokens: int = 20,
        do_sample: bool = True,
        num_beams: int = 1,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
        num_return_sequences: int = 1,
        repetition_penalty: float = 1.0
    ) -> tuple[list[rdkit.Chem.rdchem.Mol], list[str]]:

    genai = ChemBERT(
        smiles=smiles,
        iterations=iterations,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        num_beams=num_beams,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_return_sequences=num_return_sequences,
        repetition_penalty=repetition_penalty
    )

    genai.config()
    genai.generate()
    l1, l2 = genai.sanitize()
    return l1, l2
