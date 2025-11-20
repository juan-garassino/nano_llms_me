import torch
import torch.nn.functional as F
from rich.console import Console

console = Console()

def system2_reasoning_arc(model, demos_in, demos_out, test_in, num_samples=8):
    """
    System-2 Reasoning for ARC:
    1. Generate N candidate programs (using temperature sampling).
    2. Execute each program on the DEMO inputs.
    3. Calculate consistency score (how well it reconstructs demo outputs).
    4. Select the best program.
    5. Apply best program to TEST input.
    """
    model.eval()
    B = test_in.size(0)
    device = test_in.device
    
    # Encode demos (same as forward_arc)
    num_demos = demos_in.size(1)
    flat_demos_in = demos_in.view(-1, 30, 30)
    flat_demos_out = demos_out.view(-1, 30, 30)
    
    # Embed/Encode
    flat_demos_in_seq = flat_demos_in.view(-1, 900)
    flat_demos_out_seq = flat_demos_out.view(-1, 900)
    
    x_in, freq_in = model.embed(flat_demos_in_seq)
    for block in model.encoder_blocks:
        x_in = block(x_in, sparse_weights=freq_in)
        
    x_out, freq_out = model.embed(flat_demos_out_seq)
    for block in model.encoder_blocks:
        x_out = block(x_out, sparse_weights=freq_out)
        
    feat_in = x_in.mean(dim=1).view(B, num_demos, -1)
    feat_out = x_out.mean(dim=1).view(B, num_demos, -1)
    
    # Program Synthesis Input
    demo_pairs = torch.cat([feat_in, feat_out], dim=-1)
    demo_encodings = model.program_synth.demo_encoder(demo_pairs)
    program_rep = demo_encodings.mean(dim=1) # (B, 128)
    
    # Sample N programs
    # We need to repeat program_rep N times
    # (B*N_samples, 128)
    program_rep_expanded = program_rep.repeat_interleave(num_samples, dim=0)
    
    # Synthesize programs with temperature
    programs = model.program_synth.synthesize_program(program_rep_expanded, temperature=1.0)
    # programs is list of (probs, stop) tuples. Length = max_len.
    # Each element has batch size B*N_samples.
    
    # Verify on Demos
    # We need to execute these programs on the demo inputs and compare to demo outputs.
    # freq_in: (B*num_demos, 900, n_freqs). 
    # We need to expand this to (B*num_demos*num_samples, ...) ?
    # This gets expensive.
    
    # Alternative: Just verify on the first demo for efficiency?
    # Or just trust the "Consistency" of the program log-probability?
    # The user's System-2 example used "Self-Consistency" (majority vote) or "Best Logprob".
    # Let's use "Best Logprob" (confidence) if we can't easily execute.
    # But wait, we CAN execute.
    
    # Let's try to execute on the FIRST demo for each batch item to score.
    # demo 0 input freq: freq_in.view(B, num_demos, ...)[:, 0] -> (B, 900, n_freqs)
    demo0_freq = freq_in.view(B, num_demos, 900, -1)[:, 0]
    demo0_target = flat_demos_out.view(B, num_demos, 30, 30)[:, 0]
    
    # Expand for samples
    demo0_freq_exp = demo0_freq.repeat_interleave(num_samples, dim=0) # (B*S, 900, F)
    
    # Execute
    transformed_freq = model.program_synth.execute_program(demo0_freq_exp, programs)
    
    # Decode
    transformed_feat = torch.matmul(transformed_freq, model.embed.freq_basis)
    x = transformed_feat
    for block in model.decoder_blocks:
        x = block(x, sparse_weights=transformed_freq)
    x = model.norm(x)
    logits = model.head(x) # (B*S, 900, 11)
    
    # Calculate Loss against target
    demo0_target_exp = demo0_target.repeat_interleave(num_samples, dim=0).view(-1, 900)
    loss = F.cross_entropy(logits.view(-1, 11), demo0_target_exp.view(-1), reduction='none')
    loss = loss.view(B*num_samples, 900).mean(dim=1) # (B*S)
    

# -----------------------------------------------------------------------------
# Advanced Inference Strategies (System-2, Active Inference, TTC, Speculative)
# -----------------------------------------------------------------------------

@torch.no_grad()
def generate_adaptive(model, start_ids, max_new=64, retry_budget=2, entropy_th=2.2, device="cpu"):
    """
    Adaptive Test-Time Compute (TTC).
    Retries generation for tokens with high entropy (uncertainty).
    """
    model.eval()
    out = list(start_ids)
    for _ in range(max_new):
        # Context window handling would go here
        ids = torch.tensor([out], device=device) # Simplified, assumes fits in context
        logits = model(ids)
        if isinstance(logits, tuple): logits = logits[0]
        
        last = logits[:, -1, :]
        probs = F.softmax(last, dim=-1)
        
        # Calculate Entropy
        H = -(probs * probs.clamp_min(1e-9).log()).sum(-1).item()
        
        if H > entropy_th and retry_budget > 0:
            # High uncertainty: Resample multiple times and pick most confident
            cands = []
            confs = []
            for _ in range(3):
                nxt = torch.multinomial(probs, 1).item()
                conf = probs[0, nxt].item()
                cands.append(nxt)
                confs.append(conf)
            
            # Pick candidate with highest confidence
            best_idx = torch.tensor(confs).argmax().item()
            out.append(cands[best_idx])
            retry_budget -= 1
        else:
            # Standard sampling
            out.append(torch.multinomial(probs, 1).item())
            
    return out

@torch.no_grad()
def generate_active_inference(model, start_ids, horizon=16, particles=4, device="cpu"):
    """
    Active Inference Generation.
    Rolls out multiple 'particles' (future trajectories) and picks the next token
    that minimizes Expected Free Energy (NLL - Entropy).
    """
    model.eval()
    ids = torch.tensor([start_ids], device=device)
    logits = model(ids)
    if isinstance(logits, tuple): logits = logits[0]
    
    last = logits[:, -1, :]
    probs = F.softmax(last, dim=-1)
    
    # Consider top-k candidates for the next token
    topv, topi = torch.topk(probs, k=min(64, probs.size(-1)), dim=-1)
    cand_tokens = topi[0].tolist()
    
    scores = []
    # Evaluate each candidate
    for t in cand_tokens[:particles*4]:
        seq = start_ids + [t]
        seq_t = torch.tensor([seq], device=device)
        
        # Short rollout (1 step lookahead for simplicity here, can be deeper)
        l = model(seq_t)
        if isinstance(l, tuple): l = l[0]
        
        p = F.softmax(l[:, -1, :], dim=-1)
        H = -(p * p.clamp_min(1e-9).log()).sum(-1).item() # Entropy (Uncertainty)
        nll = -torch.log(probs[0, t].clamp_min(1e-9)).item() # NLL (Surprise)
        
        # Free Energy = NLL - beta * Entropy
        # We want to minimize NLL (maximize likelihood) and maximize Entropy (exploration/robustness)
        # Or in Active Inference: minimize Surprise + Uncertainty
        scores.append((nll - 0.1*H, t))
        
    scores.sort() # Ascending order of Free Energy
    return start_ids + [scores[0][1]]

@torch.no_grad()
def speculative_decoding(verifier, draft, start_ids, max_new=64, device="cpu"):
    """
    Speculative Decoding.
    Uses a small 'draft' model to propose tokens, and a large 'verifier' model to accept/reject.
    """
    verifier.eval()
    draft.eval()
    
    out = torch.tensor([start_ids], device=device)
    
    for _ in range(max_new):
        # 1. Draft proposes K tokens (here K=1 for simplicity)
        d_logits = draft(out)
        if isinstance(d_logits, tuple): d_logits = d_logits[0]
        d_next = torch.argmax(d_logits[:, -1, :], dim=-1, keepdim=True)
        
        # 2. Verifier checks
        v_logits = verifier(out)
        if isinstance(v_logits, tuple): v_logits = v_logits[0]
        v_probs = F.softmax(v_logits[:, -1, :], dim=-1)
        
        # Acceptance criterion (simplified: if verifier assigns > 5% prob to draft's choice)
        # In full spec decoding, we use rejection sampling.
        accept = v_probs.gather(-1, d_next).squeeze(-1) > 0.05
        
        if accept.item():
            out = torch.cat([out, d_next], dim=1)
        else:
            # Reject: Sample from verifier
            v_next = torch.multinomial(v_probs, 1)
            out = torch.cat([out, v_next], dim=1)
            
    return out[0].tolist()
    # Select best program per batch item
    loss = loss.view(B, num_samples)
    best_indices = torch.argmin(loss, dim=1) # (B,)
    
    # Now apply BEST program to TEST input
    # We need to extract the specific program steps for the best indices.
    # This is tricky with the list structure.
    # Easier: Just re-run the best one? Or index into the batch.
    
    # Let's just execute ALL on test, then select the output corresponding to best index.
    test_in_seq = test_in.view(B, 900)
    _, freq_test = model.embed(test_in_seq) # (B, 900, F)
    freq_test_exp = freq_test.repeat_interleave(num_samples, dim=0)
    
    test_transformed_freq = model.program_synth.execute_program(freq_test_exp, programs)
    
    test_transformed_feat = torch.matmul(test_transformed_freq, model.embed.freq_basis)
    x_test = test_transformed_feat
    for block in model.decoder_blocks:
        x_test = block(x_test, sparse_weights=test_transformed_freq)
    x_test = model.norm(x_test)
    test_logits = model.head(x_test) # (B*S, 900, 11)
    
    # Select best outputs
    test_logits = test_logits.view(B, num_samples, 900, 11)
    # gather
    best_indices_exp = best_indices.view(B, 1, 1, 1).expand(-1, -1, 900, 11)
    final_logits = torch.gather(test_logits, 1, best_indices_exp).squeeze(1)
    
    return final_logits
