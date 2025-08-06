# 🚀 QUICK FIX - Copy this code into a new cell in your notebook

def pipeline_fixed(model, tokenizer, k=10):
    """FIXED generator - replaces the broken pipeline"""
    def generate(prompt, max_len=200, min_len=50, temperature=0.8, stop_on_repeat=True):
        model.eval()
        with torch.no_grad():
            x = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(model.config.device)
            original_len = x.size(1)
            last_tokens = []
            repeat_count = 0
            
            for i in range(max_len):
                if x.size(1) >= model.config.block_size:
                    x = x[:, -model.config.block_size:]
                
                logits, _ = model(x)
                last_token_logits = logits[:, -1, :] / temperature
                next_token = top_k_sample_fixed(last_token_logits, k)
                
                # Stop on repetition
                token_val = next_token.item()
                if stop_on_repeat:
                    if len(last_tokens) > 0 and token_val == last_tokens[-1]:
                        repeat_count += 1
                        if repeat_count > 5:
                            break
                    else:
                        repeat_count = 0
                    last_tokens.append(token_val)
                    if len(last_tokens) > 10:
                        last_tokens.pop(0)
                
                x = torch.cat([x, next_token], dim=1)
                
                # Stop on sentence endings
                if i >= min_len:
                    current_text = tokenizer.decode(x[0, original_len:].tolist())
                    if any(ending in current_text[-3:] for ending in ['. ', '! ', '? ', '\n\n']):
                        break
            
            generated_tokens = x[0, original_len:].tolist()
            generated_text = tokenizer.decode(generated_tokens)
            return prompt + generated_text
    return generate

def top_k_sample_fixed(logits, k=50):
    """FIXED top-k sampling"""
    k = min(k, logits.size(-1))
    values, indices = torch.topk(logits, k, dim=-1)
    probs = torch.softmax(values, dim=-1)
    next_token_idx = torch.multinomial(probs, num_samples=1)
    next_token = indices.gather(-1, next_token_idx)
    return next_token

# TEST THE FIX:
generator = pipeline_fixed(model, tokenizer, k=15)
print("Testing fixed generator:")
result = generator('To be or not to be', max_len=150, temperature=0.8)
print(f"Result: {result}")
print(f"Length: {len(result)} characters")