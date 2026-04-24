import torch
from huggingface_hub import login
from transformers import AutoTokenizer, EsmModel
from transformers import EsmForProteinFolding

import esm
from esm.sdk.api import ESMProtein, LogitsConfig
from esm.models.esmc import ESMC
from esm.sdk.forge import ESM3ForgeInferenceClient


class ESMBaseWrapper:
    def __init__(self, device, wanted_layer=-1):
        """
        Args:
            device: 'cuda' or 'cpu'
            wanted_layer: The integer index of the layer to extract.
                          -1 for the last layer.
                          For ESM2-650M (33 layers), 0 is the embedding, 1-33 are blocks.
        """
        self.device = device
        self.wanted_layer = wanted_layer

    def get_embedding(self, sequence):
        raise NotImplementedError


class ESM2Wrapper(ESMBaseWrapper):
    def __init__(self, exact_model, device, wanted_layer=-1):
        super().__init__(device, wanted_layer)

        model_map = {
            "esm2_15B": esm.pretrained.esm2_t48_15B_UR50D,
            "esm2_3B": esm.pretrained.esm2_t36_3B_UR50D,
            "esm2_650M": esm.pretrained.esm2_t33_650M_UR50D,
            "esm2_150M": esm.pretrained.esm2_t30_150M_UR50D,
            "esm2_35M": esm.pretrained.esm2_t12_35M_UR50D,
            "esm2_8M": esm.pretrained.esm2_t6_8M_UR50D,
        }

        if exact_model in model_map:
            self.model, self.alphabet = model_map[exact_model]()
        else:
            raise ValueError(f"Unknown ESM2 model: {exact_model}")

        self.model = self.model.to(self.device).eval()
        self.batch_converter = self.alphabet.get_batch_converter()

        # Resolve negative index to actual layer number
        # self.model.num_layers gives the total count (e.g. 33)
        if self.wanted_layer == -1:
            self.target_layer = self.model.num_layers
        else:
            self.target_layer = self.wanted_layer

    def get_embedding(self, sequence):
        data = [("id", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)

        with torch.no_grad():
            # ESM2 (fairseq) expects repr_layers as a list of integers
            results = self.model(batch_tokens, repr_layers=[self.target_layer])

        return (
            results["representations"][self.target_layer][0, 1 : len(sequence) + 1]
            .cpu()
            .float()
            .numpy()
        )


class HFESM2Wrapper(ESMBaseWrapper):
    def __init__(self, exact_model, device, wanted_layer=-1):
        super().__init__(device, wanted_layer)

        model_map = {
            "esm2_15B": "facebook/esm2_t48_15B_UR50D",
            "esm2_3B": "facebook/esm2_t36_3B_UR50D",
            "esm2_650M": "facebook/esm2_t33_650M_UR50D",
            "esm2_150M": "facebook/esm2_t30_150M_UR50D",
            "esm2_35M": "facebook/esm2_t12_35M_UR50D",
            "esm2_8M": "facebook/esm2_t6_8M_UR50D",
        }

        if exact_model not in model_map:
            raise ValueError(f"Unknown ESM2 model: {exact_model}")

        model_id = model_map[exact_model]
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = (
            EsmModel.from_pretrained(model_id, torch_dtype=dtype).to(self.device).eval()
        )

    def get_embedding(self, sequence):
        inputs = self.tokenizer(sequence, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # output_hidden_states=True is required to see intermediate layers
            outputs = self.model(**inputs, output_hidden_states=True)

        # outputs.hidden_states is a tuple of (embeddings, layer_1, ..., layer_N)
        # If wanted_layer is -1, it automatically picks the last one.
        # If wanted_layer is 33, it picks the 33rd item.
        # Note: HF includes embeddings at index 0, so "Layer 1" is hidden_states[1].
        # Use simple python indexing.

        embeddings = outputs.hidden_states[self.wanted_layer][0, 1 : len(sequence) + 1]
        return embeddings.cpu().float().numpy()


from transformers import EsmTokenizer, EsmForMaskedLM


class SaProtWrapper(ESMBaseWrapper):
    def __init__(self, model_path, device, wanted_layer=-1):
        super().__init__(device, wanted_layer)

        # SaProt is based on ESM-2 but often loaded via EsmForMaskedLM
        self.tokenizer = EsmTokenizer.from_pretrained(model_path)

        # Optimized precision loading
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.model = (
            EsmForMaskedLM.from_pretrained(model_path, torch_dtype=dtype)
            .to(self.device)
            .eval()
        )

    def get_embedding(self, sequence):
        """
        Extracts embeddings for a SaProt sequence.
        Note: SaProt sequences typically look like 'A#BqC#',
        where each residue is followed by a structural digit/character.
        """
        formatted_seq = "".join([f"{r}#" for r in sequence])
        inputs = self.tokenizer(formatted_seq, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # We use output_hidden_states=True to access the encoder layers
            outputs = self.model(**inputs, output_hidden_states=True)

        # In HuggingFace ESM implementations:
        # hidden_states[0] = Input Embeddings
        # hidden_states[1...N] = Transformer Layers

        # Extract the specific layer
        # sequence_output shape: [batch, tokens, hidden_size]

        layer_output = outputs.hidden_states[self.wanted_layer][0]

        # Slicing logic for the sequence track:
        # SaProt's tokenizer handles combined tokens (e.g., 'M#').
        # We skip index 0 (BOS) and take up to the length of the tokenized sequence.
        embeddings = layer_output[1 : len(sequence) + 1]

        return embeddings.cpu().float().numpy()


class ESM3Wrapper(ESMBaseWrapper):
    def __init__(
        self, exact_model, device, wanted_layer=-1, token_file="data/hg_token.txt"
    ):
        from esm.models.esm3 import ESM3
        from huggingface_hub import login

        super().__init__(device, wanted_layer)

        if exact_model == "esm3_1B":
            model_id = "esm3-open"
        else:
            raise ValueError(f"Model name '{exact_model}' not known.")

        try:
            with open(token_file, "r") as f:
                hf_token = f.read().strip()
            login(token=hf_token)
        except FileNotFoundError:
            print(f"Warning: Token file not found at {token_file}")

        self.model = ESM3.from_pretrained(model_id).to(self.device).eval()

    def get_embedding(self, sequence=None):

        protein = ESMProtein(sequence=sequence)
        encoded_protein = self.model.encode(protein)
        seq = encoded_protein.sequence.to(self.device)
        forward_kwargs = {"sequence_tokens": seq.unsqueeze(0) if seq.ndim == 1 else seq}

        capture = {}

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                capture["embeddings"] = output[0].detach()
            else:
                capture["embeddings"] = output.detach()

        # Dynamic layer selection
        target_layer_idx = self.wanted_layer if self.wanted_layer != -1 else -1
        target_layer = self.model.transformer.blocks[target_layer_idx]

        handle = target_layer.register_forward_hook(hook_fn)
        dtype = next(self.model.parameters()).dtype

        try:
            with torch.no_grad():
                with torch.autocast(device_type=self.device, dtype=dtype):
                    self.model.forward(**forward_kwargs)
        finally:
            handle.remove()

        full_embeddings = capture["embeddings"]
        length = len(protein.sequence)
        residue_embeddings = full_embeddings[0, 1 : length + 1]
        return residue_embeddings.cpu().to(torch.float32).numpy()


class ESMFoldWrapper:
    def __init__(self, device, trunk_layer=-1):
        """
        Initializes ESMFold with the ability to target specific folding blocks.

        Args:
            device (str): Hardware device.
            trunk_layer (int): The specific block in the folding trunk (0-47).
                               Default is -1 (the final refinement block).
        """
        self.device = device
        self.trunk_layer = trunk_layer

        # Load model and tokenizer using settings from protein_folding.py
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
        self.model = EsmForProteinFolding.from_pretrained(
            "facebook/esmfold_v1", low_cpu_mem_usage=True
        )

        # Apply performance optimizations
        if "cuda" in self.device:
            # self.model.esm = self.model.esm.half()
            # torch.backends.cuda.matmul.allow_tf32 = True
            self.model.trunk.set_chunk_size(64)

        self.model.to(self.device)
        self.model.eval()

        self._captured_s_track = None

    def _hook_fn(self, module, input, output):
        """
        Captures only the sequence track (s) from the targeted folding block.
        In ESMFold, the trunk blocks return a tuple where the first element is 's'.
        """
        # output is (s, z, ...) - we only take the sequence track 's'
        self._captured_s_track = output[0]

    def get_embedding(self, sequence):
        """
        Runs the model and extracts structure-aware embeddings from the chosen trunk layer.
        """
        inputs = self.tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False
        ).to(self.device)

        # Register hook on the specific block within the trunk
        # ESMFold has 48 blocks (0 through 47)
        target_block = self.model.trunk.blocks[self.trunk_layer]
        handle = target_block.register_forward_hook(self._hook_fn)

        with torch.no_grad():
            self.model(**inputs)

        handle.remove()

        # Return the L x 1024 sequence track, removing batch dimension
        return self._captured_s_track.squeeze(0).cpu()


class ESMForgeWrapper(ESMBaseWrapper):
    def __init__(self, model_name, token_path, wanted_layer=-1):
        print("Init forge wrapper")
        # Forge API usually only returns the final layer.
        if wanted_layer != -1:
            self.return_hidden_states = True
        else:
            self.return_hidden_states = False
        self.wanted_layer = wanted_layer

        super().__init__("cpu", wanted_layer)
        with open(token_path, "r") as f:
            token = f.read().strip()
        self.client = ESM3ForgeInferenceClient(
            model=model_name, url="https://forge.evolutionaryscale.ai", token=token
        )

    def get_embedding(self, sequence):
        if len(sequence) > 2048:
            return torch.zeros((len(sequence), 2560))
        protein = ESMProtein(sequence=sequence)
        t = self.client.encode(protein)
        out = self.client.logits(
            t,
            LogitsConfig(
                sequence=False,
                return_embeddings=True,
                return_hidden_states=self.return_hidden_states,
                ith_hidden_layer=self.wanted_layer,
            ),
        )
        return out.embeddings[0, 1:-1, :].to(torch.float32).cpu().numpy()


def get_model_wrapper(model_type, device, wanted_layer=-1, token_path=None):
    if model_type == "esmfold":
        return ESMFoldWrapper(device, wanted_layer)
    elif model_type.startswith("esm2_"):
        # Use HuggingFace wrapper by default for these strings in your previous logic
        # OR switch to ESM2Wrapper if you prefer Fairseq.
        # Based on your previous snippet, you had both. Let's assume HF for now:
        return HFESM2Wrapper(model_type, device, wanted_layer)
    elif model_type == "esmc-6b-2024-12" or model_type in [
        "esm3-large-2024-03",
        "esm3-medium-2024-08",
    ]:
        return ESMForgeWrapper(model_type, token_path, wanted_layer)
    elif model_type == "esm3_1B":
        return ESM3Wrapper(model_type, device, wanted_layer)
    elif model_type in [
        "westlake-repl/SaProt_35M_AF2",
        "westlake-repl/SaProt_650M_PDB",
        "westlake-repl/SaProt_650M_AF2",
        "westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI",
    ]:
        return SaProtWrapper(model_type, device, wanted_layer)

    raise ValueError(f"Unknown model: {model_type}")
