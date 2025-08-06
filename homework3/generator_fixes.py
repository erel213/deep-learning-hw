"""
🔧 SHAKESPEARE GENERATOR FIXES
===============================

This file contains the fixed generator functions to replace the broken pipeline in your notebook.

ISSUE: The original pipeline was generating only one character because:
1. It ran for exactly max_len=1024 iterations (too many)
2. No stopping conditions 
3. No repetition detection
4. No temperature control

SOLUTION: Use the functions below in your notebook.
"""

import torch
import torch.nn.functional as F

def pipeline_fixed(model, tokenizer, k=10):
    """
    FIXED version of the text generation pipeline.
    
    Args:
        model: The trained GPT2 model
        tokenizer: The character tokenizer
        k: Number of top tokens to sample from (default: 10)
    
    Returns:
        A generate function that creates Shakespeare-like text
    """
    def generate(prompt, max_len=200, min_len=50, temperature=0.8, stop_on_repeat=True):
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
                
                # Extract logits for the last token and apply temperature
                last_token_logits = logits[:, -1, :] / temperature
                
                # Apply top k sampling
                next_token = top_k_sample_fixed(last_token_logits, k)
                
                # Check for repetition (stop if same token repeated too many times)
                token_val = next_token.item()
                if stop_on_repeat:
                    if len(last_tokens) > 0 and token_val == last_tokens[-1]:
                        repeat_count += 1
                        if repeat_count > 5:  # Stop if same token repeated 5 times
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
                if i >= min_len:
                    # Stop on period, exclamation, or question mark followed by space
                    current_text = tokenizer.decode(x[0, original_len:].tolist())
                    if any(ending in current_text[-3:] for ending in ['. ', '! ', '? ', '\n\n']):
                        break
            
            # Decode only the generated part (excluding the original prompt)
            generated_tokens = x[0, original_len:].tolist()
            generated_text = tokenizer.decode(generated_tokens)
            return prompt + generated_text
    
    return generate


def top_k_sample_fixed(logits, k=50):
    """
    FIXED version of top-k sampling function with better handling.
    
    Args:
        logits: Model output logits (tensor)
        k: Number of top tokens to consider
        
    Returns:
        Sampled token index
    """
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


def test_generator_quick(model, tokenizer):
    """
    Quick test function to verify the generator works.
    """
    print("="*60)
    print("🎭 SHAKESPEARE GENERATOR - QUICK TEST")
    print("="*60)
    
    try:
        # Create the fixed generator
        generator = pipeline_fixed(model, tokenizer, k=15)
        
        # Test with a few prompts
        test_prompts = [
            "To be or not to be",
            "All the world's a stage", 
            "Romeo, Romeo, wherefore art thou"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\n🎭 Test {i}: '{prompt}'")
            print("-" * 40)
            
            result = generator(
                prompt, 
                max_len=100, 
                min_len=30,
                temperature=0.8,
                stop_on_repeat=True
            )
            
            print(f"Result: {result}")
            print(f"Length: {len(result)} chars")
            
        print("\n" + "="*60)
        print("✅ If you see longer text above (not just 1 char), it's working!")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


def debug_step_by_step(model, tokenizer, prompt="To be", steps=5):
    """
    Debug function to see what the model predicts step by step.
    Useful for understanding why generation might fail.
    """
    print(f"\n🔍 STEP-BY-STEP DEBUG: '{prompt}'")
    print("="*50)
    
    model.eval()
    with torch.no_grad():
        x = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(model.config.device)
        print(f"Starting tokens: {x[0].tolist()}")
        
        for step in range(steps):
            # Get predictions
            logits, _ = model(x)
            last_logits = logits[:, -1, :]
            
            # Show top 5 predictions
            values, indices = torch.topk(last_logits, 5)
            probs = torch.softmax(values, dim=-1)
            
            print(f"\nStep {step + 1}:")
            print(f"  Current text: '{tokenizer.decode(x[0].tolist())}'")
            print(f"  Top 5 next predictions:")
            
            for i, (idx, prob) in enumerate(zip(indices[0], probs[0])):
                char = tokenizer.decode([idx.item()])
                print(f"    {i+1}. '{char}' (prob: {prob.item():.3f})")
            
            # Sample and continue
            next_token = top_k_sample_fixed(last_logits, k=5)
            next_char = tokenizer.decode([next_token.item()])
            print(f"  → Selected: '{next_char}'")
            
            x = torch.cat([x, next_token], dim=1)
        
        final_text = tokenizer.decode(x[0].tolist())
        print(f"\nFinal result: '{final_text}'")
        return final_text


# ================================================================
# USAGE INSTRUCTIONS
# ================================================================

"""
HOW TO USE THESE FIXES IN YOUR NOTEBOOK:

1. Copy the functions above into a new cell in your notebook

2. Replace your generator test with:
   
   # Test the fixed generator
   generator = pipeline_fixed(model, tokenizer, k=15)
   result = generator('To be or not to be', max_len=150, temperature=0.8)
   print(result)

3. For quick testing, use:
   
   test_generator_quick(model, tokenizer)

4. For debugging issues, use:
   
   debug_step_by_step(model, tokenizer, "To be", steps=5)

5. Key parameters you can adjust:
   - k: Number of top tokens to sample (5-50, higher = more diverse)
   - max_len: Maximum characters to generate (50-500)
   - min_len: Minimum before checking stop conditions (20-100) 
   - temperature: Randomness (0.1=focused, 2.0=creative)
   - stop_on_repeat: Whether to stop on repetitive text
"""