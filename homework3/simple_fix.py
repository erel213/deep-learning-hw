# 🚀 SIMPLE FIX - Maintains original interface, no temperature

def pipeline_fixed(model, tokenizer, k=10):
    """
    Fixed pipeline that maintains the original interface.
    Only fixes the core issue: stopping conditions and repetition detection.
    """
    def generate(prompt, max_len=1024):
        model.eval()
        with torch.no_grad():
            # Encode the prompt
            x = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(model.config.device)
            original_len = x.size(1)
            
            # Track for repetition detection
            last_tokens = []
            repeat_count = 0
            
            # Generate tokens
            for i in range(max_len):
                # Limit sequence length to block_size for efficiency
                if x.size(1) >= model.config.block_size:
                    x = x[:, -model.config.block_size:]
                
                # Get model predictions
                logits, _ = model(x)
                
                # Extract logits for the last token (NO temperature applied)
                last_token_logits = logits[:, -1, :]
                
                # Apply top k sampling
                next_token = top_k_sample_fixed(last_token_logits, k)
                
                # Check for repetition (stop if same token repeated too many times)
                token_val = next_token.item()
                if len(last_tokens) > 0 and token_val == last_tokens[-1]:
                    repeat_count += 1
                    if repeat_count > 8:  # Stop if same token repeated 8 times
                        break
                else:
                    repeat_count = 0
                
                # Keep track of last few tokens
                last_tokens.append(token_val)
                if len(last_tokens) > 10:
                    last_tokens.pop(0)
                
                # Append the new token
                x = torch.cat([x, next_token], dim=1)
                
                # Stop if we've generated a reasonable amount and hit certain characters
                if i >= 50:  # Minimum 50 characters before checking stop conditions
                    current_text = tokenizer.decode(x[0, original_len:].tolist())
                    if any(ending in current_text[-3:] for ending in ['. ', '! ', '? ', '\n\n']):
                        break
            
            # Decode and return the generated text (including original prompt)
            generated_tokens = x[0].tolist()
            generated_text = tokenizer.decode(generated_tokens)
            return generated_text
    
    return generate


def top_k_sample_fixed(logits, k=50):
    """Fixed top-k sampling function"""
    # Ensure k doesn't exceed vocabulary size
    k = min(k, logits.size(-1))
    
    # Get top k values and indices
    values, indices = torch.topk(logits, k, dim=-1)
    
    # Apply softmax to get probabilities
    probs = torch.softmax(values, dim=-1)
    
    # Sample from the top-k distribution
    next_token_idx = torch.multinomial(probs, num_samples=1)
    
    # Use gather to select the actual token index
    next_token = indices.gather(-1, next_token_idx)
    
    return next_token


# REPLACE YOUR ORIGINAL PIPELINE WITH THIS:
# Just change this line in your notebook:
# OLD: generator = pipeline(model, tokenizer, k=10)
# NEW: generator = pipeline_fixed(model, tokenizer, k=10)

# Test with original interface:
generator = pipeline_fixed(model, tokenizer, k=10)
print("Testing with original interface:")
result = generator('To be or not to be', max_len=200)  # Original interface: just prompt and max_len
print(f"Result: {result}")
print(f"Length: {len(result)} characters")