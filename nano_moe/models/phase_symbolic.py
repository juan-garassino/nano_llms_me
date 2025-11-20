import torch
import torch.nn as nn
import torch.nn.functional as F
from .phase import SFPTBlock, SparseFourierEmbedding

# ====================== Phase Library ======================
class PhaseOperator(nn.Module):
    """
    A symbolic operation represented as phase transformation
    Examples:
    - Rotate 90°: specific phase pattern in 2D Fourier space
    - Reflect: phase inversion pattern
    - Extend: amplify low frequencies
    - Fill: suppress high frequencies
    """
    def __init__(self, n_freqs=256, dim=128, name="unknown"):
        super().__init__()
        self.name = name
        
        # Learnable phase transformation (complex rotation in Fourier space)
        self.theta_real = nn.Parameter(torch.randn(n_freqs, dim) * 0.01)
        self.theta_imag = nn.Parameter(torch.randn(n_freqs, dim) * 0.01)
        
        # Amplitude modulation (which frequencies to amplify/suppress)
        self.amplitude = nn.Parameter(torch.ones(n_freqs) * 0.5)
    
    def forward(self, freq_codes):
        """
        freq_codes: (B, L, n_freqs) — sparse frequency activation
        Returns: transformed frequency codes
        """
        B, L, n_freqs = freq_codes.shape
        
        # Phase rotation in frequency space
        # This is like: exp(i*theta) * freq_codes (complex multiplication)
        cos_theta = torch.cos(self.theta_real.mean(dim=-1))  # (n_freqs,)
        sin_theta = torch.sin(self.theta_imag.mean(dim=-1))
        
        # Apply phase rotation + amplitude modulation
        amp = torch.sigmoid(self.amplitude)  # (n_freqs,)
        
        # Real-valued approximation of complex rotation
        # Note: freq_codes are real (magnitudes or sparse activations), so we modulate them.
        # If freq_codes were complex, we'd do full complex mult.
        # Here we treat them as magnitudes and modulate them by the "real part" of the phase shift + amp.
        transformed = freq_codes * cos_theta.unsqueeze(0).unsqueeze(0) * amp.unsqueeze(0).unsqueeze(0)
        
        return transformed

class PhaseSymbolicLibrary(nn.Module):
    """
    Library of symbolic operations as phase transformations
    Each operation is a learnable phase operator
    """
    def __init__(self, n_ops=16, n_freqs=256, dim=128):
        super().__init__()
        self.n_ops = n_ops
        
        # Library of phase operators (learned symbolic primitives)
        self.operators = nn.ModuleList([
            PhaseOperator(n_freqs, dim, name=f"op_{i}")
            for i in range(n_ops)
        ])
        
        # Identity operator (no-op)
        self.identity = PhaseOperator(n_freqs, dim, name="identity")
        with torch.no_grad():
            self.identity.theta_real.zero_()
            self.identity.theta_imag.zero_()
            self.identity.amplitude.fill_(0.0) # Sigmoid(0) = 0.5. Maybe fill with large value for 1.0?
            # Let's leave as is for now, or set to inverse sigmoid(1.0)
    
    def forward(self, freq_codes, op_idx):
        """
        Apply operator op_idx to freq_codes
        op_idx: (B,) — which operator to apply (discrete or soft)
        """
        if isinstance(op_idx, int):
            # Hard selection
            return self.operators[op_idx](freq_codes)
        else:
            # Soft mixture of operators (differentiable)
            # op_idx is (B, n_ops) probabilities
            result = torch.zeros_like(freq_codes)
            for i, op in enumerate(self.operators):
                weight = op_idx[:, i].view(-1, 1, 1)  # (B, 1, 1)
                result = result + weight * op(freq_codes)
            return result

# ====================== Phase Program Synthesizer ======================
class PhaseProgramSynthesizer(nn.Module):
    """
    Learns to compose phase operators into programs
    Input: demo examples (input→output pairs)
    Output: sequence of phase operators that implements the rule
    """
    def __init__(self, n_freqs=256, dim=128, n_ops=16, max_program_len=5):
        super().__init__()
        self.n_ops = n_ops
        self.max_program_len = max_program_len
        
        # Library of symbolic operations
        self.library = PhaseSymbolicLibrary(n_ops, n_freqs, dim)
        
        # Program encoder: demos → program representation
        self.demo_encoder = nn.Sequential(
            nn.Linear(dim * 2, 256),  # input + output concatenated
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )
        
        # Program decoder: representation → operator sequence
        self.program_decoder = nn.LSTM(128, 128, num_layers=2, batch_first=True)
        
        # Operator selector: LSTM hidden → operator probabilities
        self.op_selector = nn.Linear(128, n_ops)
        
        # Termination predictor
        self.termination = nn.Linear(128, 1)
    
    def encode_demos(self, demo_inputs, demo_outputs):
        """
        Encode demonstration examples into program representation
        demo_inputs: list of (B, L, dim) tensors
        demo_outputs: list of (B, L, dim) tensors
        """
        # Concatenate input-output pairs
        demos = []
        for inp, out in zip(demo_inputs, demo_outputs):
            # Average pool over spatial dimension
            inp_avg = inp.mean(dim=1)  # (B, dim)
            out_avg = out.mean(dim=1)
            demos.append(torch.cat([inp_avg, out_avg], dim=-1))  # (B, dim*2)
        
        # Encode each demo
        demo_encodings = [self.demo_encoder(d) for d in demos]
        
        # Aggregate (mean pooling over demos)
        program_rep = torch.stack(demo_encodings).mean(dim=0)  # (B, 128)
        
        return program_rep
    
    def synthesize_program(self, program_rep, temperature=1.0):
        """
        Generate operator sequence from program representation
        Returns: list of (operator_probs, should_terminate) tuples
        """
        B = program_rep.size(0)
        h = program_rep.unsqueeze(1)  # (B, 1, 128)
        
        program = []
        hidden = None
        
        for step in range(self.max_program_len):
            # Decode one step
            out, hidden = self.program_decoder(h, hidden)  # out: (B, 1, 128)
            
            # Select operator
            op_logits = self.op_selector(out.squeeze(1))  # (B, n_ops)
            op_probs = F.softmax(op_logits / temperature, dim=-1)
            
            # Check termination
            term_logit = self.termination(out.squeeze(1))  # (B, 1)
            should_stop = torch.sigmoid(term_logit) > 0.5
            
            program.append((op_probs, should_stop))
            
            # If all batch items want to stop, terminate
            if should_stop.all():
                break
            
            # Feed output back as input for next step
            h = out
        
        return program
    
    def execute_program(self, freq_codes, program):
        """
        Execute synthesized program on input frequency codes
        freq_codes: (B, L, n_freqs)
        program: list of (op_probs, should_stop)
        """
        x = freq_codes
        
        for op_probs, should_stop in program:
            # Apply soft mixture of operators
            x_new = self.library(x, op_probs)
            
            # Mask out stopped examples
            mask = (~should_stop).float().view(-1, 1, 1)
            x = x * (1 - mask) + x_new * mask
        
        return x

class HybridPhaseSymbolicARC(nn.Module):
    """
    Hybrid architecture:
    1. SFPT extracts patterns → sparse frequency codes
    2. Program synthesizer learns symbolic rules as phase operators
    3. Execute program on test input
    4. Decode back to grid
    """
    def __init__(self, vocab_size, dim, n_layer, n_heads, n_freqs, top_k, expansion, dropout, n_ops=16, max_program_len=5):
        super().__init__()
        
        # Pattern extraction (from SFPT)
        self.embed = SparseFourierEmbedding(vocab_size, n_freqs, top_k, dim)
        
        self.encoder_blocks = nn.ModuleList([
            SFPTBlock(dim, n_heads, expansion, dropout)
            for _ in range(n_layer // 2)
        ])
        
        # Program synthesis
        self.program_synth = PhaseProgramSynthesizer(
            n_freqs, dim, n_ops=n_ops, max_program_len=max_program_len
        )
        
        # Decoder (frequency codes → output grid)
        self.decoder_blocks = nn.ModuleList([
            SFPTBlock(dim, n_heads, expansion, dropout)
            for _ in range(n_layer // 2)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
    
    def forward_arc(self, demos_in, demos_out, test_in):
        """
        Forward pass for ARC meta-learning loop.
        demos_in: (B, num_demos, H, W)
        demos_out: (B, num_demos, H, W)
        test_in: (B, H, W)
        """
        B = test_in.size(0)
        
        # Encode demos
        # We need to process each demo.
        # demos_in is (B, N, 30, 30)
        # We can flatten B*N to process in parallel
        num_demos = demos_in.size(1)
        
        flat_demos_in = demos_in.view(-1, 30, 30) # (B*N, 30, 30)
        flat_demos_out = demos_out.view(-1, 30, 30)
        
        # Embed and Encode
        # Assuming input is indices (0-10)
        # We need to flatten to (B*N, 900) for the embedding if it expects sequence
        # Or keep 2D if embedding handles it.
        # Our SparseFourierEmbedding expects (B, L) indices.
        flat_demos_in_seq = flat_demos_in.view(-1, 900)
        flat_demos_out_seq = flat_demos_out.view(-1, 900)
        
        # Encode inputs
        x_in, freq_in = self.embed(flat_demos_in_seq)
        for block in self.encoder_blocks:
            x_in = block(x_in, sparse_weights=freq_in)
            
        # Encode outputs
        x_out, freq_out = self.embed(flat_demos_out_seq)
        for block in self.encoder_blocks:
            x_out = block(x_out, sparse_weights=freq_out)
            
        # Reshape back to (B, N, dim)
        # We pool over the sequence (900 pixels) to get a vector per demo?
        # The synthesizer expects (B, dim) per demo.
        # Let's mean pool.
        feat_in = x_in.mean(dim=1).view(B, num_demos, -1)
        feat_out = x_out.mean(dim=1).view(B, num_demos, -1)
        
        # Synthesize Program
        # We need to pass list of (B, dim) to synthesizer? 
        # The synthesizer `encode_demos` took list of tensors.
        # Let's adapt it or just pass the tensor (B, N, dim).
        
        # Let's update synthesizer to take (B, N, dim)
        # For now, let's just loop or reshape to match what synthesizer expects
        # Synthesizer expects: demo_inputs: list of (B, L, dim) ... wait, my implementation of encode_demos:
        # "for inp, out in zip(demo_inputs, demo_outputs):"
        # It expects lists of tensors.
        
        # Let's just pass the features to a new method in synthesizer or adapt here.
        # I'll implement a helper here.
        
        # Concatenate in/out: (B, N, 2*dim)
        demo_pairs = torch.cat([feat_in, feat_out], dim=-1)
        
        # Encode each pair: (B, N, 256) -> (B, N, 128)
        # We can use the synthesizer's encoder
        demo_encodings = self.program_synth.demo_encoder(demo_pairs)
        
        # Mean pool over demos: (B, 128)
        program_rep = demo_encodings.mean(dim=1)
        
        # Generate program
        program = self.program_synth.synthesize_program(program_rep)
        
        # Process Test Input
        test_in_seq = test_in.view(B, 900)
        x_test, freq_test = self.embed(test_in_seq)
        
        # Execute Program on Test Frequencies
        # freq_test is (B, 900, n_freqs)
        transformed_freq = self.program_synth.execute_program(freq_test, program)
        
        # Decode
        # We use the transformed frequencies to modulate the decoder?
        # Or we use the transformed frequencies AS the input to decoder?
        # In `phase.py`, the embedding returns `x` (projected) and `sparse_weights` (freqs).
        # The blocks use `sparse_weights` to modulate attention.
        # So we should use `x_test` (content) but modulated by `transformed_freq` (style/rule)?
        # OR, we should reconstruct `x` from `transformed_freq`.
        # The user concept: "Reconstruct features from transformed frequencies: transformed_feat = torch.matmul(transformed_freq, self.embed.freq_basis)"
        
        transformed_feat = torch.matmul(transformed_freq, self.embed.freq_basis)
        
        # Decoder
        x = transformed_feat
        for block in self.decoder_blocks:
            x = block(x, sparse_weights=transformed_freq)
            
        x = self.norm(x)
        logits = self.head(x)
        
        return logits
